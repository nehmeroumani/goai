package openaicompat

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// --- N1: hoisted helpers ---

func TestBoolPtr(t *testing.T) {
	p := BoolPtr(true)
	if p == nil || *p != true {
		t.Fatalf("BoolPtr(true) = %v", p)
	}
	q := BoolPtr(false)
	if q == nil || *q != false {
		t.Fatalf("BoolPtr(false) = %v", q)
	}
	// Distinct pointers so mutations cannot alias.
	if p == q {
		t.Error("BoolPtr must allocate a fresh pointer per call")
	}
}

func TestMergeHeaders(t *testing.T) {
	user := map[string]string{"X-User": "u", "X-Both": "user-wins"}
	fixed := map[string]string{"X-Fixed": "f", "X-Both": "fixed"}
	got := MergeHeaders(user, fixed)

	if got["X-User"] != "u" {
		t.Errorf("X-User = %q", got["X-User"])
	}
	if got["X-Fixed"] != "f" {
		t.Errorf("X-Fixed = %q", got["X-Fixed"])
	}
	if got["X-Both"] != "user-wins" {
		t.Errorf("X-Both = %q, want user-wins (user overlays fixed)", got["X-Both"])
	}
	// Inputs must not be mutated.
	if fixed["X-Both"] != "fixed" {
		t.Error("fixed input was mutated")
	}
	if user["X-Both"] != "user-wins" {
		t.Error("user input was mutated")
	}
}

// --- FINDING-004: per-request headers cannot override the configured auth ---

func TestSanitizeRequestHeaders_DropsAuthorization(t *testing.T) {
	got := sanitizeRequestHeaders(map[string]string{
		"Authorization": "Bearer attacker",
		"authorization": "lower",
		"AUTHORIZATION": "upper",
		"X-Keep":        "kept",
	})
	if _, ok := got["Authorization"]; ok {
		t.Error("Authorization must be dropped")
	}
	if _, ok := got["authorization"]; ok {
		t.Error("authorization must be dropped (case-insensitive)")
	}
	if _, ok := got["AUTHORIZATION"]; ok {
		t.Error("AUTHORIZATION must be dropped (case-insensitive)")
	}
	if got["X-Keep"] != "kept" {
		t.Errorf("X-Keep = %q, want kept", got["X-Keep"])
	}
}

func TestNewChatModel_PerRequestAuthCannotOverride(t *testing.T) {
	var gotAuth string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return okJSONResponse(), nil
	})
	m := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("real-key"),
		HTTPClient:  &http.Client{Transport: tr},
	})
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Headers:  map[string]string{"Authorization": "Bearer attacker"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-key" {
		t.Errorf("Authorization = %q, want configured 'Bearer real-key' (auth must win)", gotAuth)
	}
}

// --- FINDING-003: bounded accumulation ---

func TestParseStream_ToolCallArgsOverCap(t *testing.T) {
	orig := maxToolCallArgsBytes
	maxToolCallArgsBytes = 8
	defer func() { maxToolCallArgsBytes = orig }()

	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"0123456789"}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var gotErr error
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotErr = chunk.Error
		}
	}
	if gotErr == nil {
		t.Fatal("expected ChunkError when tool-call args exceed the cap")
	}
	if !strings.Contains(gotErr.Error(), "tool call arguments exceed") {
		t.Errorf("error = %v", gotErr)
	}
}

func TestParseStream_CitationsOverCap(t *testing.T) {
	orig := maxCitationsBytes
	maxCitationsBytes = 4
	defer func() { maxCitationsBytes = orig }()

	input := `data: {"citations":["https://example.com/a-long-url"]}
data: {"choices":[{"delta":{"content":"hi"},"index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var gotErr error
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotErr = chunk.Error
		}
	}
	if gotErr == nil {
		t.Fatal("expected ChunkError when citations exceed the cap")
	}
	if !strings.Contains(gotErr.Error(), "citations exceed") {
		t.Errorf("error = %v", gotErr)
	}
}

func TestReadResponseBody_OverCap(t *testing.T) {
	orig := maxResponseBodyBytes
	maxResponseBodyBytes = 4
	defer func() { maxResponseBodyBytes = orig }()

	_, err := ReadResponseBody(strings.NewReader("0123456789"))
	if err == nil {
		t.Fatal("expected error when response body exceeds the cap")
	}
	if !strings.Contains(err.Error(), "response body exceeds") {
		t.Errorf("error = %v", err)
	}
}

func TestReadResponseBody_UnderCap(t *testing.T) {
	orig := maxResponseBodyBytes
	maxResponseBodyBytes = 100
	defer func() { maxResponseBodyBytes = orig }()

	data, err := ReadResponseBody(strings.NewReader("small"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "small" {
		t.Errorf("data = %q", data)
	}
}

func TestReadResponseBody_ReadError(t *testing.T) {
	_, err := ReadResponseBody(errReader{})
	if err == nil || !strings.Contains(err.Error(), "read boom") {
		t.Fatalf("expected read error, got %v", err)
	}
}

// --- FINDING-003 integration: DoGenerate surfaces the cap error ---

func TestNewChatModel_ResponseBodyOverCap(t *testing.T) {
	orig := maxResponseBodyBytes
	maxResponseBodyBytes = 8
	defer func() { maxResponseBodyBytes = orig }()

	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)),
		}, nil
	})
	m := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		HTTPClient:  &http.Client{Transport: tr},
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err == nil || !strings.Contains(err.Error(), "response body exceeds") {
		t.Fatalf("expected cap error, got %v", err)
	}
}
