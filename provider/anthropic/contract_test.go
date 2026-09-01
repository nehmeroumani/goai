package anthropic

// Contract tests for the full-provider-audit fixes (docs/reviews/full-provider-audit.md,
// items #10-#19). Each test asserts the FULL outgoing request body (golden JSON) and/or
// the feature-gated anthropic-beta header, so that a schema drift (extra/missing/misnamed
// field, or a beta header leaking onto requests that do not need it) is caught.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// mustCaptureRequest runs DoGenerate against a mock server and returns the decoded
// outgoing request body plus the anthropic-beta header. The mock returns a minimal
// valid Messages response so DoGenerate completes.
func mustCaptureRequest(t *testing.T, params provider.GenerateParams) (map[string]any, string) {
	t.Helper()
	var bodyStr string
	var beta string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodyStr = string(b)
		beta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "text/event-stream")
		if strings.Contains(bodyStr, `"stream":true`) {
			// Thinking models use the streaming transport (see useStreamingTransport);
			// the mock must speak SSE so accumulateStreamedMessage can reassemble it.
			_, _ = fmt.Fprint(w, `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(server.Close)

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := model.DoGenerate(t.Context(), params); err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(bodyStr), &body); err != nil {
		t.Fatalf("unmarshal captured body %q: %v", bodyStr, err)
	}
	return body, beta
}

// assertGoldenBody compares the captured body against a golden JSON document. Go's
// encoding/json sorts map keys deterministically, so the comparison is order-independent.
func assertGoldenBody(t *testing.T, got map[string]any, wantJSON string) {
	t.Helper()
	var want map[string]any
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatalf("invalid golden JSON: %v", err)
	}
	gotJSON, _ := json.Marshal(got)
	wantJSONNorm, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSONNorm) {
		t.Errorf("request body mismatch\n  got:  %s\n  want: %s", gotJSON, wantJSONNorm)
	}
}

// #12 -- tool_choice "none" must be sent as {"type":"none"} WHILE the tools array is
// kept registered (the old behaviour deleted tools). Golden full-body assertion.
func TestContract_ToolChoiceNone_KeepsToolsRegistered(t *testing.T) {
	body, _ := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get the weather", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: "none",
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": false,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"tools": [{"name":"get_weather","description":"Get the weather","input_schema":{"type":"object"}}],
		"tool_choice": {"type":"none"}
	}`)
}

// #13 -- the newly-added thinking-capable models must advertise Reasoning capability.
func TestContract_NewThinkingModels_AdvertiseReasoning(t *testing.T) {
	for _, id := range []string{"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"} {
		m := Chat(id, WithAPIKey("k")).(*chatModel)
		if !m.Capabilities().Reasoning {
			t.Errorf("%s: Capabilities().Reasoning = false, want true", id)
		}
	}
}

// #14 -- context_management must produce the full body AND gate the context-1m beta.
func TestContract_ContextManagement_FullBodyAndBeta(t *testing.T) {
	body, beta := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"contextManagement": map[string]any{
				"edits": []any{
					map[string]any{"type": "clear_tool_uses_20250919", "clearAtLeast": float64(50000)},
				},
			},
		},
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": false,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"context_management": {"edits":[{"type":"clear_tool_uses_20250919","clear_at_least":50000}]}
	}`)
	if !strings.Contains(beta, betaContextManagement) {
		t.Errorf("anthropic-beta = %q, want it to contain %q", beta, betaContextManagement)
	}
}

// #15 -- speed must produce the full body AND gate the fast-mode beta.
func TestContract_Speed_FullBodyAndBeta(t *testing.T) {
	body, beta := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"speed": "fast",
		},
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": false,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"speed": "fast"
	}`)
	if !strings.Contains(beta, betaFastMode) {
		t.Errorf("anthropic-beta = %q, want it to contain %q", beta, betaFastMode)
	}
}

// #16 -- a plain Messages request must carry EXACTLY the baseline beta header, with no
// claude-code-20250219 leaking in.
func TestContract_PlainRequest_BetaIsExactlyBaseline(t *testing.T) {
	_, beta := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if beta != betaFeatures {
		t.Errorf("anthropic-beta = %q, want exactly %q (no claude-code)", beta, betaFeatures)
	}
}

// #17 -- a RemoteRef image must emit a file-backed image block AND gate the files beta.
func TestContract_RemoteRefImage_FullBodyAndFilesBeta(t *testing.T) {
	body, beta := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, RemoteRef: &provider.RemoteFileRef{ID: "file_img_1"}},
		}}},
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": false,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"file_img_1"}}]}]
	}`)
	if !strings.Contains(beta, filesBetaHeader) {
		t.Errorf("anthropic-beta = %q, want it to contain %q", beta, filesBetaHeader)
	}
}

// #18 -- thinking enabled + forced tool_choice must downgrade to auto in the full body.
func TestContract_ThinkingForcedToolChoice_DowngradesToAuto(t *testing.T) {
	body, _ := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get the weather", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: "required",
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "enabled", "budgetTokens": float64(2000)},
		},
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": true,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"tools": [{"name":"get_weather","description":"Get the weather","input_schema":{"type":"object"}}],
		"tool_choice": {"type":"auto"},
		"thinking": {"type":"enabled","budget_tokens":2000}
	}`)
}

// #19 -- web_search_20260318 gates the code-execution-web-tools-2026-03-18 beta and
// serializes its options; code_execution_20250522 adds no beta.
func TestContract_ServerToolBetas_WebSearch20260318AndCodeExecution20250522(t *testing.T) {
	body, beta := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools:    []provider.ToolDefinition{Tools.WebSearch_20260318(WithResponseInclusion("low"))},
	})

	assertGoldenBody(t, body, `{
		"model": "claude-sonnet-4-20250514",
		"stream": false,
		"max_tokens": 16384,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"tools": [{"type":"web_search_20260318","name":"web_search","response_inclusion":"low"}]
	}`)
	if !strings.Contains(beta, "code-execution-web-tools-2026-03-18") {
		t.Errorf("anthropic-beta = %q, want it to contain code-execution-web-tools-2026-03-18", beta)
	}

	// code_execution_20250522 is GA -- no extra beta beyond the baseline.
	_, beta2 := mustCaptureRequest(t, provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools:    []provider.ToolDefinition{Tools.CodeExecution_20250522()},
	})
	if strings.Contains(beta2, "code-execution") {
		t.Errorf("anthropic-beta = %q, code_execution_20250522 must not add a beta", beta2)
	}
}

// #10/#11 -- golden multipart upload request: exactly one form part named "file", and
// no unofficial "purpose" or "expires_in_seconds" fields.
func TestContract_FileUpload_GoldenMultipartRequest(t *testing.T) {
	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_golden","type":"file","size_bytes":3,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"a.txt"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()
	if _, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("abc"),
		Filename:  "a.txt",
		MediaType: "text/plain",
		Purpose:   "assistants", // must be ignored
	}); err != nil {
		t.Fatalf("UploadFile: %v", err)
	}

	if strings.Contains(body, "purpose") {
		t.Errorf("upload body contains a purpose field: %q", body)
	}
	if strings.Contains(body, "expires_in_seconds") {
		t.Errorf("upload body contains expires_in_seconds unexpectedly: %q", body)
	}
	if c := strings.Count(body, "name=\"file\""); c != 1 {
		t.Errorf("found %d file form parts, want exactly 1: %q", c, body)
	}
}

// #10 -- response direction: the Files API returns created_at as an RFC3339 string and
// size_bytes/mime_type; the parsed result must reflect the official fields.
func TestContract_FileUpload_ResponseParsesOfficialFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_rfc3339","type":"file","size_bytes":1234,"mime_type":"image/png","created_at":"2026-08-30T12:00:00Z","filename":"img.png"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()
	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:   strings.NewReader("png-bytes"),
		Filename: "img.png",
	})
	if err != nil {
		t.Fatalf("UploadFile: %v", err)
	}
	if ref.ID != "file_rfc3339" {
		t.Errorf("ref.ID = %q, want file_rfc3339", ref.ID)
	}
	// mime_type from the API wins over content sniffing.
	if ref.MediaType != "image/png" {
		t.Errorf("ref.MediaType = %q, want image/png (from mime_type)", ref.MediaType)
	}
}
