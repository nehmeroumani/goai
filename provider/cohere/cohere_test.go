package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("texts = %v, want [Hello]", texts)
	}
}

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat" {
			t.Errorf("path = %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != false {
			t.Error("expected stream=false")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":5}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want 'Hello world'", result.Text)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"message":"Rate limited"}`)
	}))
	defer server.Close()

	model := Chat("model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNoTokenSource(t *testing.T) {
	model := Chat("model")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithAPIKey("key"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.opts.httpClient != c {
		t.Error("custom client not set")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithAPIKey("k"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("command-r-plus", WithAPIKey("k"))
	if model.ModelID() != "command-r-plus" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("path = %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_document" {
			t.Errorf("input_type = %v", req["input_type"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2,0.3],[0.4,0.5,0.6]]},"meta":{"billed_units":{"input_tokens":10}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Errorf("embeddings count = %d", len(result.Embeddings))
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
}

func TestEmbeddingModelID(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"))
	if model.ModelID() != "embed-v4.0" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestEmbeddingMaxValues(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"))
	if model.MaxValuesPerCall() != 96 {
		t.Errorf("MaxValuesPerCall() = %d", model.MaxValuesPerCall())
	}
}

func TestEmbeddingNoTokenSource(t *testing.T) {
	model := Embedding("embed-v4.0")
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dynamic" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("dynamic")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChatWithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools = %d, want 1", len(tools))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[],"tool_calls":[{"id":"tc1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: []byte(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "get_weather" {
		t.Errorf("ToolName = %q", result.ToolCalls[0].Name)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

// errReader is an io.ReadCloser that returns an error on Read.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, fmt.Errorf("forced read error") }
func (errReader) Close() error             { return nil }

type errBodyTransport struct{}

func (t *errBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       errReader{},
		Header:     make(http.Header),
	}, nil
}

func TestDoGenerate_ReadError(t *testing.T) {
	model := Chat("m",
		WithAPIKey("k"),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestDoGenerate_ParseError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{invalid json`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestStream_ToolCallEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"city\\\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\": \\\"Paris\\\"}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotToolStart bool
	var gotToolCall bool
	var gotFinish bool
	var toolDeltas []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCallStreamStart {
			gotToolStart = true
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
			if chunk.ToolCallID != "tc1" {
				t.Errorf("ToolCallID = %q", chunk.ToolCallID)
			}
		}
		if chunk.Type == provider.ChunkToolCallDelta {
			toolDeltas = append(toolDeltas, chunk)
		}
		if chunk.Type == provider.ChunkToolCall {
			gotToolCall = true
			if chunk.ToolCallID != "tc1" {
				t.Errorf("ToolCallID = %q", chunk.ToolCallID)
			}
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
			if chunk.ToolInput != `{"city": "Paris"}` {
				t.Errorf("ToolInput = %q, want %q", chunk.ToolInput, `{"city": "Paris"}`)
			}
		}
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishToolCalls {
				t.Errorf("FinishReason = %q", chunk.FinishReason)
			}
		}
	}

	wantDeltas := []string{`{"city"`, `: "Paris"}`}
	if len(toolDeltas) != len(wantDeltas) {
		t.Fatalf("expected %d ChunkToolCallDelta, got %d", len(wantDeltas), len(toolDeltas))
	}
	for i, want := range wantDeltas {
		if toolDeltas[i].ToolCallID != "tc1" {
			t.Errorf("delta[%d].ToolCallID = %q, want %q", i, toolDeltas[i].ToolCallID, "tc1")
		}
		if toolDeltas[i].ToolName != "get_weather" {
			t.Errorf("delta[%d].ToolName = %q, want %q", i, toolDeltas[i].ToolName, "get_weather")
		}
		if toolDeltas[i].ToolInput != want {
			t.Errorf("delta[%d].ToolInput = %q, want %q", i, toolDeltas[i].ToolInput, want)
		}
	}
	if !gotToolStart {
		t.Error("no tool call start chunk")
	}
	if !gotToolCall {
		t.Error("no tool call chunk with accumulated arguments")
	}
	if !gotFinish {
		t.Error("no finish chunk")
	}
}

func TestStream_ContextCancellation(t *testing.T) {
	// Test context cancellation - with TrySend, the goroutine exits cleanly
	// instead of sending an error chunk (which could block if buffer is full).
	ctx, cancel := context.WithCancel(t.Context())

	out := make(chan provider.StreamChunk, 64)
	data := "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n"
	cancel() // cancel before parsing

	done := make(chan struct{})
	go func() {
		parseChatStream(ctx, strings.NewReader(data), out)
		close(done)
	}()
	<-done

	// Goroutine exited without blocking - drain any chunks.
	for range out {
	}
}

func TestStream_ScannerError(t *testing.T) {
	out := make(chan provider.StreamChunk, 64)
	go parseChatStream(t.Context(), errReader{}, out)

	var gotError bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotError = true
			if !strings.Contains(chunk.Error.Error(), "reading stream") {
				t.Errorf("unexpected error: %s", chunk.Error)
			}
		}
	}
	if !gotError {
		t.Error("expected error chunk from scanner error")
	}
}

func TestStream_MaxTokensFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"truncated\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"MAX_TOKENS\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":100}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishLength {
				t.Errorf("FinishReason = %q, want length", chunk.FinishReason)
			}
		}
	}
	if !gotFinish {
		t.Error("no finish chunk")
	}
}

func TestStream_UnknownFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"UNKNOWN_REASON\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.FinishReason != provider.FinishOther {
				t.Errorf("FinishReason = %q, want other", chunk.FinishReason)
			}
		}
	}
}

func TestStream_BilledUnitsFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// No "tokens" field -- should fall back to billed_units.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"billed_units\":{\"input_tokens\":15,\"output_tokens\":25}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.Usage.InputTokens != 15 {
				t.Errorf("InputTokens = %d, want 15", chunk.Usage.InputTokens)
			}
			if chunk.Usage.OutputTokens != 25 {
				t.Errorf("OutputTokens = %d, want 25", chunk.Usage.OutputTokens)
			}
		}
	}
}

func TestStream_DONE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		// This should not be reached after [DONE].
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"after done\"}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "hi" {
		t.Errorf("texts = %v, want [hi]", texts)
	}
}

func TestStream_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {invalid json}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"after\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Invalid JSON lines should be skipped; valid ones should still work.
	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "after" {
		t.Errorf("texts = %v, want [after]", texts)
	}
}

func TestStream_NonDataLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: ping\n\n")
		_, _ = fmt.Fprint(w, ": comment\n\n")
		_, _ = fmt.Fprint(w, "\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "ok" {
		t.Errorf("texts = %v, want [ok]", texts)
	}
}

func TestGenerate_ToolCallFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[],"tool_calls":[{"id":"tc1","type":"function","function":{"name":"calc","arguments":"{\"x\":1}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":5,"output_tokens":10}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

func TestGenerate_MaxTokensFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"truncated"}]},"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":5,"output_tokens":100}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishLength {
		t.Errorf("FinishReason = %q, want length", result.FinishReason)
	}
}

func TestGenerate_UnknownFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"UNKNOWN","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishOther {
		t.Errorf("FinishReason = %q, want other", result.FinishReason)
	}
}

func TestGenerate_BilledUnitsFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// tokens is zero, should fall back to billed_units.
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":15,"output_tokens":25},"tokens":{"input_tokens":0,"output_tokens":0}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens != 15 {
		t.Errorf("InputTokens = %d, want 15", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 25 {
		t.Errorf("OutputTokens = %d, want 25", result.Usage.OutputTokens)
	}
}

func TestGenerate_MultipleContentBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want 'Hello world'", result.Text)
	}
}

func TestGenerate_EmptyContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":0}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
}

func TestBuildChatRequest_ToolResultMessages(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
			{Role: provider.RoleAssistant, Content: []provider.Part{
				{Type: provider.PartToolCall, ToolCallID: "tc1", ToolName: "calc", ToolInput: json.RawMessage(`{"x":1}`)},
			}},
			{Role: "tool", Content: []provider.Part{
				{Type: provider.PartToolResult, ToolCallID: "tc1", ToolOutput: "42"},
			}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}
	// Third message should be tool result.
	if msgs[2]["role"] != "tool" {
		t.Errorf("role = %v", msgs[2]["role"])
	}
	if msgs[2]["tool_call_id"] != "tc1" {
		t.Errorf("tool_call_id = %v", msgs[2]["tool_call_id"])
	}
	if msgs[2]["content"] != "42" {
		t.Errorf("content = %v", msgs[2]["content"])
	}
}

func TestBuildChatRequest_ToolCallInAssistant(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleAssistant, Content: []provider.Part{
				{Type: provider.PartToolCall, ToolCallID: "tc1", ToolName: "calc", ToolInput: json.RawMessage(`{"x":1}`)},
			}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	tcs, _ := msgs[0]["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls = %d, want 1", len(tcs))
	}
	// When there are tool calls but no text, content should be empty string.
	if msgs[0]["content"] != "" {
		t.Errorf("content = %v, want empty string", msgs[0]["content"])
	}
}

func TestBuildChatRequest_SystemMessage(t *testing.T) {
	params := provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}
	body := buildChatRequest(params, "model", false)
	msgs, _ := body["messages"].([]map[string]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0]["role"] != "system" {
		t.Errorf("first msg role = %v", msgs[0]["role"])
	}
}

func TestBuildChatRequest_WithParams(t *testing.T) {
	temp := 0.5
	topP := 0.9
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		MaxOutputTokens: 100,
		Temperature:     &temp,
		TopP:            &topP,
	}
	body := buildChatRequest(params, "model", false)
	if body["max_tokens"] != 100 {
		t.Errorf("max_tokens = %v", body["max_tokens"])
	}
	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v", body["temperature"])
	}
	if body["p"] != 0.9 {
		t.Errorf("p = %v", body["p"])
	}
}

func TestBuildChatRequest_InvalidToolSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "t", Description: "d", InputSchema: []byte(`{invalid}`)},
		},
	}
	body := buildChatRequest(params, "model", false)
	tools, _ := body["tools"].([]map[string]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	// Invalid schema should fallback to empty map.
	fn, _ := tools[0]["function"].(map[string]any)
	schema, ok := fn["parameters"].(map[string]any)
	if !ok || len(schema) != 0 {
		t.Errorf("parameters = %v, want empty map", fn["parameters"])
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"message":"bad request"}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_ReadError(t *testing.T) {
	model := Embedding("embed-v4.0",
		WithAPIKey("k"),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ParseError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{invalid json}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ConnectionError(t *testing.T) {
	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Embedding("embed-v4.0", WithAPIKey("k"), WithHTTPClient(c))
	em := model.(*embeddingModel)
	if em.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_AuthHeaderWinsOverWithHeaders(t *testing.T) {
	// A caller-supplied WithHeaders header named like the auth header must
	// NOT override the real credential on the DoEmbed path.
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0",
		WithAPIKey("real-key"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"Authorization": "Bearer spoofed-key"}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-key" {
		t.Errorf("Authorization = %q, want %q (credential must win over WithHeaders)", gotAuth, "Bearer real-key")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- BF.16: Tool call delta accumulation ---

func TestStream_ToolCallDeltaAccumulation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Start with initial args fragment.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"x\\\"\"}}}}}\n\n")
		// Delta fragments.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\": 1, \"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"\\\"y\\\": 2}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":10}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolInput string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolInput = chunk.ToolInput
		}
	}
	if toolInput != `{"x": 1, "y": 2}` {
		t.Errorf("accumulated ToolInput = %q, want %q", toolInput, `{"x": 1, "y": 2}`)
	}
}

func TestStream_ToolCallNullArgs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"noop\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"null\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "noop"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			if chunk.ToolInput != "{}" {
				t.Errorf("ToolInput = %q, want %q (null should become empty object)", chunk.ToolInput, "{}")
			}
		}
	}
}

func TestStream_MultipleToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// First tool call.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc1\",\"type\":\"function\",\"function\":{\"name\":\"add\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"a\\\": 1}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		// Second tool call.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc2\",\"type\":\"function\",\"function\":{\"name\":\"mul\",\"arguments\":\"\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"b\\\": 2}\"}}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-call-end\"}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":10}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	if len(toolCalls) != 2 {
		t.Fatalf("got %d tool calls, want 2", len(toolCalls))
	}
	if toolCalls[0].ToolCallID != "tc1" || toolCalls[0].ToolName != "add" {
		t.Errorf("first tool call: id=%q name=%q", toolCalls[0].ToolCallID, toolCalls[0].ToolName)
	}
	if toolCalls[1].ToolCallID != "tc2" || toolCalls[1].ToolName != "mul" {
		t.Errorf("second tool call: id=%q name=%q", toolCalls[1].ToolCallID, toolCalls[1].ToolName)
	}
}

// --- BF.16: Thinking / reasoning support ---

func TestBuildChatRequest_Thinking(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type":         "enabled",
				"budgetTokens": 4096,
			},
		},
	}
	body := buildChatRequest(params, "command-a-reasoning", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v", thinking["type"])
	}
	if thinking["token_budget"] != 4096 {
		t.Errorf("thinking.token_budget = %v", thinking["token_budget"])
	}
}

func TestBuildChatRequest_ThinkingDefaultType(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"budgetTokens": 2048,
			},
		},
	}
	body := buildChatRequest(params, "command-a-reasoning", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want 'enabled' (default)", thinking["type"])
	}
}

func TestBuildChatRequest_ThinkingDisabled(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type": "disabled",
			},
		},
	}
	body := buildChatRequest(params, "model", false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not set in request body")
	}
	if thinking["type"] != "disabled" {
		t.Errorf("thinking.type = %v", thinking["type"])
	}
}

func TestBuildChatRequest_NoThinking(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}
	body := buildChatRequest(params, "model", false)
	if _, ok := body["thinking"]; ok {
		t.Error("thinking should not be set when not in ProviderOptions")
	}
}

func TestGenerate_ThinkingContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"thinking","thinking":"Let me reason about this..."},{"type":"text","text":"The answer is 42."}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}`)
	}))
	defer server.Close()

	model := Chat("command-a-reasoning", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "The answer is 42." {
		t.Errorf("Text = %q", result.Text)
	}
	// Reasoning should be in ProviderMetadata.
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	reasoning, ok := result.ProviderMetadata["cohere"]["reasoning"].(string)
	if !ok || reasoning != "Let me reason about this..." {
		t.Errorf("reasoning = %q", reasoning)
	}
}

func TestStream_ThinkingContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Thinking content block.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"thinking\",\"thinking\":\"\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"thinking\":\"Reasoning...\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-end\",\"index\":0}\n\n")
		// Text content block.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-start\",\"index\":1,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":1,\"delta\":{\"message\":{\"content\":{\"text\":\"Answer\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-end\",\"index\":1}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotReasoning bool
	var gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning {
			gotReasoning = true
			if chunk.Text != "Reasoning..." {
				t.Errorf("reasoning text = %q", chunk.Text)
			}
		}
		if chunk.Type == provider.ChunkText {
			gotText = true
			if chunk.Text != "Answer" {
				t.Errorf("text = %q", chunk.Text)
			}
		}
	}
	if !gotReasoning {
		t.Error("no reasoning chunk")
	}
	if !gotText {
		t.Error("no text chunk")
	}
}

// --- BF.16: Embedding input_type and truncate ---

func TestEmbedding_InputType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_query" {
			t.Errorf("input_type = %v, want search_query", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":5}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"query"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "search_query",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("embeddings = %d", len(result.Embeddings))
	}
}

func TestEmbedding_InputTypeClassification(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "classification" {
			t.Errorf("input_type = %v, want classification", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "classification",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_InputTypeClustering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "clustering" {
			t.Errorf("input_type = %v, want clustering", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "clustering",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_DefaultInputType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_document" {
			t.Errorf("input_type = %v, want search_document (default)", req["input_type"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_Truncate(t *testing.T) {
	for _, tc := range []string{"NONE", "START", "END"} {
		t.Run(tc, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var req map[string]any
				_ = json.Unmarshal(body, &req)
				if req["truncate"] != tc {
					t.Errorf("truncate = %v, want %q", req["truncate"], tc)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
			}))
			defer server.Close()

			model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
			_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
				ProviderOptions: map[string]any{
					"truncate": tc,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEmbedding_NoTruncateByDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if _, ok := req["truncate"]; ok {
			t.Errorf("truncate should not be set by default, got %v", req["truncate"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_InputTypeAndTruncateCombined(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["input_type"] != "search_query" {
			t.Errorf("input_type = %v", req["input_type"])
		}
		if req["truncate"] != "END" {
			t.Errorf("truncate = %v", req["truncate"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":5}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"inputType": "search_query",
			"truncate":  "END",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- BF.16: Streaming message-end with delta.finish_reason ---

func TestStream_MessageEndDeltaFormat(t *testing.T) {
	// Vercel reference uses delta.finish_reason and delta.usage.tokens for message-end.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":8,\"output_tokens\":3}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.FinishReason != provider.FinishStop {
				t.Errorf("FinishReason = %q, want stop", chunk.FinishReason)
			}
			if chunk.Usage.InputTokens != 8 {
				t.Errorf("InputTokens = %d, want 8", chunk.Usage.InputTokens)
			}
			if chunk.Usage.OutputTokens != 3 {
				t.Errorf("OutputTokens = %d, want 3", chunk.Usage.OutputTokens)
			}
		}
	}
}

func TestChat_Generate_UnknownFinishReason(t *testing.T) {
	// When finish_reason is missing/empty, parseChatResponse should fall back to FinishOther.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// No finish_reason field → defaults to empty string → mapFinishReason("") returns ""
		_, _ = fmt.Fprint(w, `{"id":"resp-1","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishOther {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishOther)
	}
}

func TestChat_Stream_UnknownFinishReason(t *testing.T) {
	// When message-end has no finish_reason (or empty) at both event.FinishReason
	// and event.Delta.FinishReason, parseChatStream should fall back to FinishOther.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
		// message-end with no finish_reason at either level
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"delta\":{\"usage\":{\"tokens\":{\"input_tokens\":3,\"output_tokens\":1}}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkStepFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishOther {
				t.Errorf("FinishReason = %q, want %q", chunk.FinishReason, provider.FinishOther)
			}
		}
	}
	if !gotFinish {
		t.Error("no step_finish chunk received")
	}
}

func TestChat_PerRequestHeaders(t *testing.T) {
	// Verify that per-request headers (params.Headers) are extracted from the body
	// as _headers and applied to the HTTP request.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify custom header was applied.
		if got := r.Header.Get("X-Custom-Header"); got != "custom-value" {
			t.Errorf("X-Custom-Header = %q, want %q", got, "custom-value")
		}
		if got := r.Header.Get("X-Another"); got != "another-value" {
			t.Errorf("X-Another = %q, want %q", got, "another-value")
		}

		// Verify _headers is NOT in the JSON body (should be stripped).
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		_ = json.Unmarshal(body, &reqBody)
		if _, ok := reqBody["_headers"]; ok {
			t.Error("_headers should be stripped from the request body")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"test","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":1}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
			"X-Another":       "another-value",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestBuildChatRequest_HeadersInjection(t *testing.T) {
	// Verify that buildChatRequest adds _headers to the body when params.Headers is set.
	body := buildChatRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{
			"X-Test": "value",
		},
	}, "command-r-plus", false)

	headers, ok := body["_headers"].(map[string]string)
	if !ok {
		t.Fatal("_headers not found in request body")
	}
	if headers["X-Test"] != "value" {
		t.Errorf("_headers[X-Test] = %q, want %q", headers["X-Test"], "value")
	}
}

func TestBuildChatRequest_ResponseFormatWithSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "person",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}
	body := buildChatRequest(params, "command-r-plus", false)
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
	// json_schema must be the raw JSON-schema object, NOT a {name, schema} wrapper.
	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		t.Fatalf("json_schema = %T, want raw schema object", rf["json_schema"])
	}
	if js["type"] != "object" {
		t.Errorf("json_schema.type = %v, want object", js["type"])
	}
	if _, hasName := js["name"]; hasName {
		t.Error("json_schema should not contain a name key (no {name,schema} wrapper)")
	}
	if _, hasSchema := js["schema"]; hasSchema {
		t.Error("json_schema should not contain a nested schema key (no {name,schema} wrapper)")
	}
}

func TestBuildChatRequest_ResponseFormatNoSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{},
	}
	body := buildChatRequest(params, "command-r-plus", false)
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
	if _, ok := rf["json_schema"]; ok {
		t.Error("json_schema should not be set when Schema is empty")
	}
}

func TestBuildChatRequest_NoHeadersWhenEmpty(t *testing.T) {
	// Verify that buildChatRequest does NOT add _headers when params.Headers is empty.
	body := buildChatRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}, "command-r-plus", false)

	if _, ok := body["_headers"]; ok {
		t.Error("_headers should not be present when Headers is empty")
	}
}

// --- Citation tests ---

func TestChat_Generate_Citations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-1",
			"message": {
				"role": "assistant",
				"content": [{"type": "text", "text": "The answer is documented."}],
				"citations": [
					{
						"start": 0,
						"end": 10,
						"text": "The answer",
						"sources": [
							{"document": {"id": "doc-1", "title": "Reference Doc", "text": "full text"}}
						]
					},
					{
						"start": 11,
						"end": 24,
						"text": "is documented",
						"sources": [
							{"document": {"id": "doc-2", "title": "Another Doc", "text": "more text"}}
						]
					}
				]
			},
			"finish_reason": "COMPLETE",
			"usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}}
		}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Sources) != 2 {
		t.Fatalf("len(Sources) = %d, want 2", len(result.Sources))
	}

	s0 := result.Sources[0]
	if s0.Type != "document" {
		t.Errorf("Sources[0].Type = %q, want document", s0.Type)
	}
	if s0.ID != "doc-1" {
		t.Errorf("Sources[0].ID = %q, want doc-1", s0.ID)
	}
	if s0.Title != "Reference Doc" {
		t.Errorf("Sources[0].Title = %q", s0.Title)
	}
	if s0.StartIndex != 0 || s0.EndIndex != 10 {
		t.Errorf("Sources[0] indices = (%d, %d), want (0, 10)", s0.StartIndex, s0.EndIndex)
	}
	if s0.ProviderMetadata == nil {
		t.Fatal("Sources[0].ProviderMetadata is nil")
	}
	if cohere, ok := s0.ProviderMetadata["cohere"].(map[string]any); ok {
		if cohere["text"] != "The answer" {
			t.Errorf("ProviderMetadata text = %v", cohere["text"])
		}
	} else {
		t.Error("Sources[0].ProviderMetadata missing cohere key")
	}

	s1 := result.Sources[1]
	if s1.ID != "doc-2" || s1.Title != "Another Doc" {
		t.Errorf("Sources[1] = %+v", s1)
	}
}

func TestChat_Generate_NoCitations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-2",
			"message": {
				"role": "assistant",
				"content": [{"type": "text", "text": "No citations here."}]
			},
			"finish_reason": "COMPLETE",
			"usage": {"tokens": {"input_tokens": 5, "output_tokens": 3}}
		}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 0 {
		t.Errorf("len(Sources) = %d, want 0", len(result.Sources))
	}
}

func TestChat_Stream_Citations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":{\"start\":0,\"end\":5,\"text\":\"Hello\",\"sources\":[{\"document\":{\"id\":\"d1\",\"title\":\"Doc1\",\"text\":\"hello text\"}}]}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var finishChunk provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	if finishChunk.Metadata == nil {
		t.Fatal("finish chunk has no metadata")
	}
	sources, ok := finishChunk.Metadata["sources"].([]provider.Source)
	if !ok {
		t.Fatal("finish chunk metadata missing 'sources'")
	}
	if len(sources) != 1 {
		t.Fatalf("len(sources) = %d, want 1", len(sources))
	}
	if sources[0].Type != "document" || sources[0].ID != "d1" {
		t.Errorf("sources[0] = %+v", sources[0])
	}
	if sources[0].Title != "Doc1" {
		t.Errorf("sources[0].Title = %q", sources[0].Title)
	}
}

func TestResolveEnv_APIKey(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "env-key")
	m := Chat("command-r-plus")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from COHERE_API_KEY")
	}
}

func TestResolveEnv_BaseURL(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "env-key")
	t.Setenv("COHERE_BASE_URL", "https://custom.cohere.com/v2")
	m := Chat("command-r-plus")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.cohere.com/v2" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestResolveEnv_NotOverrideExplicit(t *testing.T) {
	t.Setenv("COHERE_BASE_URL", "https://env.url")
	m := Chat("command-r-plus", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestParseChatStream_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in parseChatStream.

	tests := []struct {
		name  string
		input string
	}{
		{
			// content-delta thinking (line 662)
			name:  "thinking_delta",
			input: "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"thinking\":\"hmm\"}}}}\n",
		},
		{
			// content-delta text (line 669)
			name:  "text_delta",
			input: "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hello\"}}}}\n",
		},
		{
			// tool-plan-delta (line 874)
			name:  "tool_plan_delta",
			input: "data: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"plan\"}}}}\n",
		},
		{
			// tool-call-start (line 691)
			name:  "tool_call_start",
			input: "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"t1\",\"function\":{\"name\":\"fn\"}}}}}\n",
		},
		{
			// message-end StepFinish (line 759)
			name:  "message_end",
			input: "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}}\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				parseChatStream(ctx, strings.NewReader(tc.input), out)
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested: tool-call-delta ChunkToolCallDelta TrySend cancel.
	t.Run("tool_call_delta_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"t1\",\"function\":{\"name\":\"fn\"}}}}}\n" +
			"data: {\"type\":\"tool-call-delta\",\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{}\"}}}}}\n"

		done := make(chan struct{})
		go func() {
			parseChatStream(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // tool start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: tool-call-end (line 710) requires tool-call-start TrySend to succeed first.
	t.Run("tool_call_end_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"tool-call-start\",\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"t1\",\"function\":{\"name\":\"fn\"}}}}}\n" +
			"data: {\"type\":\"tool-call-end\"}\n"

		done := make(chan struct{})
		go func() {
			parseChatStream(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // tool start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: message-end ChunkFinish (line 774) requires StepFinish TrySend to succeed first.
	t.Run("message_end_finish_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":10,\"output_tokens\":5}}}}\n"

		done := make(chan struct{})
		go func() {
			parseChatStream(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // step finish
		cancel()
		<-done
		for range out {
		}
	})
}

func TestBuildChatRequest_ToolChoiceRequired(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "my_tool", Description: "desc", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: "required",
	}
	body := buildChatRequest(params, "command-r-plus", false)
	if body["tool_choice"] != "REQUIRED" {
		t.Errorf("tool_choice = %v, want %q", body["tool_choice"], "REQUIRED")
	}
}

func TestBuildChatRequest_ToolChoiceSpecificTool(t *testing.T) {
	// Selecting a specific tool by name is unsupported by Cohere v2, so
	// tool_choice must be omitted entirely (falls back to provider default).
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ToolChoice: "my_func",
	}
	body := buildChatRequest(params, "command-r-plus", false)
	if _, ok := body["tool_choice"]; ok {
		t.Errorf("tool_choice should be omitted for a specific tool name, got %v", body["tool_choice"])
	}
}

func TestBuildChatRequest_ToolChoiceNone(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ToolChoice: "none",
	}
	body := buildChatRequest(params, "command-r-plus", false)
	if body["tool_choice"] != "NONE" {
		t.Errorf("tool_choice = %v, want %q", body["tool_choice"], "NONE")
	}
}

func TestBuildChatRequest_ToolChoiceAuto(t *testing.T) {
	// "auto" is the Cohere default, so tool_choice is omitted.
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ToolChoice: "auto",
	}
	body := buildChatRequest(params, "command-r-plus", false)
	if _, ok := body["tool_choice"]; ok {
		t.Errorf("tool_choice should be omitted for auto, got %v", body["tool_choice"])
	}
}

func TestBuildChatRequest_StopSequences(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		StopSequences: []string{"STOP", "END"},
	}
	body := buildChatRequest(params, "command-r-plus", false)
	ss, ok := body["stop_sequences"].([]string)
	if !ok {
		t.Fatalf("stop_sequences not a []string, got %T", body["stop_sequences"])
	}
	if len(ss) != 2 || ss[0] != "STOP" || ss[1] != "END" {
		t.Errorf("stop_sequences = %v, want [STOP END]", ss)
	}
}

func TestEmbedding_MultiValueOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return embeddings in the same positional order as the inputs.
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]},"meta":{"billed_units":{"input_tokens":3}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"first", "second", "third"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 3 {
		t.Fatalf("embeddings count = %d, want 3", len(result.Embeddings))
	}
	// Index 0 → [1,0,0], Index 1 → [0,1,0], Index 2 → [0,0,1].
	if result.Embeddings[0][0] != 1.0 || result.Embeddings[0][1] != 0.0 {
		t.Errorf("embeddings[0] = %v, want [1,0,0]", result.Embeddings[0])
	}
	if result.Embeddings[1][1] != 1.0 || result.Embeddings[1][0] != 0.0 {
		t.Errorf("embeddings[1] = %v, want [0,1,0]", result.Embeddings[1])
	}
	if result.Embeddings[2][2] != 1.0 || result.Embeddings[2][0] != 0.0 {
		t.Errorf("embeddings[2] = %v, want [0,0,1]", result.Embeddings[2])
	}
}

func TestMapFinishReason_ERROR(t *testing.T) {
	if got := mapFinishReason("ERROR"); got != provider.FinishError {
		t.Errorf("mapFinishReason(ERROR) = %q, want %q", got, provider.FinishError)
	}
}

func TestMapFinishReason_Empty(t *testing.T) {
	if got := mapFinishReason(""); got != "" {
		t.Errorf("mapFinishReason(\"\") = %q, want empty", got)
	}
}

func TestMapFinishReason_Unknown(t *testing.T) {
	if got := mapFinishReason("SOMETHING_ELSE"); got != provider.FinishOther {
		t.Errorf("mapFinishReason(SOMETHING_ELSE) = %q, want %q", got, provider.FinishOther)
	}
}

func TestEmbedding_ResponseModelPopulated(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2,0.3]]},"meta":{"billed_units":{"input_tokens":5}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Response.Model != "embed-v4.0" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "embed-v4.0")
	}
}

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to the Cohere
// provider succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(server.URL))

	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoGenerate unexpected error: %v", err)
	}
	if result.Text != "ok" {
		t.Errorf("DoGenerate Text = %q, want ok", result.Text)
	}

	streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"ok\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n")
	}))
	defer streamServer.Close()

	streamModel := Chat("command-r-plus", WithAPIKey("test-key"), WithBaseURL(streamServer.URL))

	streamResult, err := streamModel.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoStream unexpected error: %v", err)
	}
	var texts []string
	for chunk := range streamResult.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "ok" {
		t.Errorf("DoStream texts = %v, want [ok]", texts)
	}
}

// --- #40: k / seed / frequency_penalty / presence_penalty ---

func TestBuildChatRequest_SamplingParams(t *testing.T) {
	topK := 40
	seed := 12345
	freq := 0.5
	presence := 0.2
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		TopK:             &topK,
		Seed:             &seed,
		FrequencyPenalty: &freq,
		PresencePenalty:  &presence,
	}
	body := buildChatRequest(params, "command-r-plus", false)
	if body["k"] != 40 {
		t.Errorf("k = %v, want 40", body["k"])
	}
	if body["seed"] != 12345 {
		t.Errorf("seed = %v, want 12345", body["seed"])
	}
	if body["frequency_penalty"] != 0.5 {
		t.Errorf("frequency_penalty = %v, want 0.5", body["frequency_penalty"])
	}
	if body["presence_penalty"] != 0.2 {
		t.Errorf("presence_penalty = %v, want 0.2", body["presence_penalty"])
	}
}

func TestBuildChatRequest_SamplingParamsUnset(t *testing.T) {
	body := buildChatRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}, "command-r-plus", false)
	for _, key := range []string{"k", "seed", "frequency_penalty", "presence_penalty"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s should not be set when nil", key)
		}
	}
}

// --- #41: tool_plan ---

func TestGenerate_ToolPlan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","tool_plan":"First I will look up the weather.","content":[],"tool_calls":[{"id":"tc1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":10,"output_tokens":20}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	md := result.ProviderMetadata["cohere"]
	if md == nil {
		t.Fatal("ProviderMetadata['cohere'] is nil")
	}
	if got, _ := md["tool_plan"].(string); got != "First I will look up the weather." {
		t.Errorf("tool_plan = %q", got)
	}
}

func TestGenerate_NoToolPlan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderMetadata != nil {
		if md := result.ProviderMetadata["cohere"]; md != nil {
			if _, ok := md["tool_plan"]; ok {
				t.Error("tool_plan should not be set when absent")
			}
		}
	}
}

func TestStream_ToolPlanDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-plan-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"I will check the \"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"tool-plan-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"weather first.\"}}}}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 2 || texts[0] != "I will check the " || texts[1] != "weather first." {
		t.Errorf("texts = %v, want [I will check the  weather first.]", texts)
	}
}

// --- #42: STOP_SEQUENCE / TIMEOUT finish reasons ---

func TestMapFinishReason_STOP_SEQUENCE(t *testing.T) {
	if got := mapFinishReason("STOP_SEQUENCE"); got != provider.FinishStop {
		t.Errorf("mapFinishReason(STOP_SEQUENCE) = %q, want %q", got, provider.FinishStop)
	}
}

func TestMapFinishReason_TIMEOUT(t *testing.T) {
	if got := mapFinishReason("TIMEOUT"); got != provider.FinishOther {
		t.Errorf("mapFinishReason(TIMEOUT) = %q, want %q", got, provider.FinishOther)
	}
}

func TestGenerate_StopSequenceFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"STOP_SEQUENCE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
}

func TestStream_TimeoutFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message-end\",\"finish_reason\":\"TIMEOUT\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n")
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
			if chunk.FinishReason != provider.FinishOther {
				t.Errorf("FinishReason = %q, want %q", chunk.FinishReason, provider.FinishOther)
			}
		}
	}
	if !gotFinish {
		t.Error("no finish chunk")
	}
}

// --- #43: embed output_dimension / embedding_types ---

func TestEmbedding_OutputDimension(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["output_dimension"] != float64(512) {
			t.Errorf("output_dimension = %v, want 512", req["output_dimension"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"output_dimension": 512,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_EmbeddingTypes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		types, ok := req["embedding_types"].([]any)
		if !ok || len(types) != 2 || types[0] != "float" || types[1] != "binary" {
			t.Errorf("embedding_types = %v, want [float binary]", req["embedding_types"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"embedding_types": []string{"float", "binary"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_NoOutputParamsByDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if _, ok := req["output_dimension"]; ok {
			t.Error("output_dimension should not be set by default")
		}
		if _, ok := req["embedding_types"]; ok {
			t.Error("embedding_types should not be set by default")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

// --- Golden full-body contract tests (#38, #39, #40, #43) ---
//
// These capture the outgoing request body through the full DoGenerate/DoEmbed
// HTTP path (httptest server) and compare the relevant subtree against the
// latest Cohere v2 schema, rather than asserting individual fields in
// buildChatRequest.

// #38: tool_choice is serialized as uppercase REQUIRED/NONE in the outgoing
// /chat request body; "auto" (the provider default) is omitted.
func TestChat_Generate_ToolChoice_GoldenBody(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice string
		want       string // golden JSON subtree, or "" meaning "must be omitted"
	}{
		{name: "required", toolChoice: "required", want: `{"tool_choice":"REQUIRED"}`},
		{name: "none", toolChoice: "none", want: `{"tool_choice":"NONE"}`},
		{name: "auto-omitted", toolChoice: "auto", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotBody, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
			}))
			defer server.Close()

			model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
			_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
				Messages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
				},
				Tools: []provider.ToolDefinition{
					{Name: "my_tool", Description: "desc", InputSchema: json.RawMessage(`{"type":"object"}`)},
				},
				ToolChoice: tt.toolChoice,
			})
			if err != nil {
				t.Fatal(err)
			}

			var req map[string]any
			if err := json.Unmarshal(gotBody, &req); err != nil {
				t.Fatalf("unmarshal request body: %v", err)
			}
			if tt.want == "" {
				if _, ok := req["tool_choice"]; ok {
					t.Errorf("tool_choice = %v, want omitted", req["tool_choice"])
				}
				return
			}
			got, _ := json.Marshal(map[string]any{"tool_choice": req["tool_choice"]})
			if string(got) != tt.want {
				t.Errorf("tool_choice subtree = %s, want %s", got, tt.want)
			}
		})
	}
}

// #39: response_format.json_schema carries the raw JSON-schema object directly
// (no {name, schema} wrapper) under type=json_object.
func TestChat_Generate_ResponseFormat_GoldenBody(t *testing.T) {
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "person",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var req map[string]any
	if err := json.Unmarshal(gotBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	want := map[string]any{
		"type": "json_object",
		"json_schema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		},
	}
	got, _ := json.Marshal(req["response_format"])
	wantJSON, _ := json.Marshal(want)
	if string(got) != string(wantJSON) {
		t.Errorf("response_format subtree = %s, want %s", got, wantJSON)
	}
}

// #40: k/seed/frequency_penalty/presence_penalty are mapped onto the outgoing
// /chat request body.
func TestChat_Generate_SamplingParams_GoldenBody(t *testing.T) {
	topK := 40
	seed := 98765
	freq := 0.5
	presence := 0.2

	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"123","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`)
	}))
	defer server.Close()

	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		TopK:             &topK,
		Seed:             &seed,
		FrequencyPenalty: &freq,
		PresencePenalty:  &presence,
	})
	if err != nil {
		t.Fatal(err)
	}

	var req map[string]any
	if err := json.Unmarshal(gotBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	keys := []string{"k", "seed", "frequency_penalty", "presence_penalty"}
	got := map[string]any{}
	for _, k := range keys {
		if v, ok := req[k]; ok {
			got[k] = v
		}
	}
	want := map[string]any{
		"k":                 float64(40),
		"seed":              float64(98765),
		"frequency_penalty": float64(0.5),
		"presence_penalty":  float64(0.2),
	}
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSON) {
		t.Errorf("sampling subtree = %s, want %s", gotJSON, wantJSON)
	}
}

// #43: embed output_dimension and embedding_types are carried on the outgoing
// /embed request body alongside the standard fields.
func TestEmbedding_OutputParams_GoldenBody(t *testing.T) {
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":1}}}`)
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"output_dimension": 512,
			"embedding_types":  []string{"float", "binary"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var req map[string]any
	if err := json.Unmarshal(gotBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	want := map[string]any{
		"model":            "embed-v4.0",
		"texts":            []any{"hello"},
		"input_type":       "search_document",
		"output_dimension": float64(512),
		"embedding_types":  []any{"float", "binary"},
	}
	gotJSON, _ := json.Marshal(req)
	wantJSON, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSON) {
		t.Errorf("embed body = %s, want %s", gotJSON, wantJSON)
	}
}

// TestChat_ResponseBodyOverCap verifies the success-path bounded read in
// DoGenerate: a response body larger than maxCohereResponseBytes is rejected.
func TestChat_ResponseBodyOverCap(t *testing.T) {
	transport := &fixedBodyTransport{body: io.NopCloser(io.LimitReader(zeroReader{}, int64(maxCohereResponseBytes+2)))}
	model := Chat("command-r-plus", WithAPIKey("k"), WithBaseURL("http://fake"), WithHTTPClient(&http.Client{Transport: transport}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for oversized response body")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want an 'exceeds' over-cap error", err)
	}
}

// TestEmbedding_ResponseBodyOverCap verifies the success-path bounded read in
// DoEmbed.
func TestEmbedding_ResponseBodyOverCap(t *testing.T) {
	transport := &fixedBodyTransport{body: io.NopCloser(io.LimitReader(zeroReader{}, int64(maxCohereResponseBytes+2)))}
	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL("http://fake"), WithHTTPClient(&http.Client{Transport: transport}))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for oversized response body")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want an 'exceeds' over-cap error", err)
	}
}

// TestEmbedding_ErrorBodyBounded verifies the error-path bounded read in
// DoEmbed: an error response body larger than maxCohereErrorBytes is truncated,
// so the tail marker never reaches the extracted error message.
func TestEmbedding_ErrorBodyBounded(t *testing.T) {
	const tailMarker = "TAIL-MARKER-SHOULD-NOT-APPEAR"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.Copy(w, io.MultiReader(
			strings.NewReader(`{"message":"`),
			io.LimitReader(zeroReader{}, int64(maxCohereErrorBytes+len(tailMarker))),
			strings.NewReader(tailMarker),
		))
	}))
	defer server.Close()

	model := Embedding("embed-v4.0", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *goai.APIError", err)
	}
	if strings.Contains(apiErr.Message, tailMarker) {
		t.Errorf("error message contains tail marker; error body was not bounded: %q", apiErr.Message)
	}
}

// zeroReader yields an endless stream of bytes without allocating.
type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'a'
	}
	return len(p), nil
}

// fixedBodyTransport returns a canned 200 response with the given body.
type fixedBodyTransport struct {
	body io.ReadCloser
}

func (t *fixedBodyTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       t.body,
		Header:     make(http.Header),
	}, nil
}
