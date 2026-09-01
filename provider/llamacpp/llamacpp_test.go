package llamacpp

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

// roundTripFunc adapts a function to the http.RoundTripper interface so tests
// can intercept and stub outgoing requests without binding a network listener.
type roundTripFunc func(r *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization header, got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"llama3","choices":[{"message":{"role":"assistant","content":"Hello from llama.cpp"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`)
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello from llama.cpp" {
		t.Errorf("Text = %q", result.Text)
	}
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("llama3", WithBaseURL(server.URL))
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
	if len(texts) != 1 || texts[0] != "Hi" {
		t.Errorf("texts = %v", texts)
	}
}

func TestChat_DefaultBaseURL(t *testing.T) {
	model := Chat("llama3")
	if model.ModelID() != "llama3" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
	// The default server exposes the OpenAI-compatible API under /v1, so the
	// default base URL must include the /v1 prefix for chat + embeddings to hit
	// the correct path (/v1/chat/completions, /v1/embeddings).
	if defaultBaseURL != "http://localhost:8080/v1" {
		t.Errorf("defaultBaseURL = %q, want %q", defaultBaseURL, "http://localhost:8080/v1")
	}
}

// TestChat_DefaultBaseURL_HitsV1Path is a REQUEST-direction contract test for
// audit item #52: llama.cpp's default server exposes the OpenAI-compatible API
// under /v1, so with no WithBaseURL the outgoing request must hit
// http://localhost:8080/v1/chat/completions (the /v1 prefix is required for
// chat + embeddings to reach the correct handler). A custom RoundTripper
// captures the real request URL without binding a network listener.
func TestChat_DefaultBaseURL_HitsV1Path(t *testing.T) {
	var gotURL string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		gotURL = r.URL.String()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"x","model":"llama3","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)),
		}, nil
	})}

	// No WithBaseURL -> the default base URL is used.
	model := Chat("llama3", WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "http://localhost:8080/v1/chat/completions"
	if gotURL != want {
		t.Errorf("request URL = %q, want %q", gotURL, want)
	}
}

// TestEmbedding_DefaultBaseURL_HitsV1Path is the embedding-side counterpart of
// the #52 contract: the default base URL must place embeddings under /v1 too
// (/v1/embeddings), not at the server root.
func TestEmbedding_DefaultBaseURL_HitsV1Path(t *testing.T) {
	var gotURL string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		gotURL = r.URL.String()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"data":[{"embedding":[0.1,0.2],"index":0}],"usage":{"prompt_tokens":3,"total_tokens":3}}`)),
		}, nil
	})}

	model := Embedding("nomic-embed-text", WithHTTPClient(client))
	if _, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{}); err != nil {
		t.Fatal(err)
	}
	want := "http://localhost:8080/v1/embeddings"
	if gotURL != want {
		t.Errorf("request URL = %q, want %q", gotURL, want)
	}
}

func TestChat_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	// Verify map is copied (caller mutation has no effect)
	headers := map[string]string{"X-Custom": "val"}
	model := Chat("m", WithBaseURL(server.URL), WithHeaders(headers))
	headers["X-Custom"] = "mutated" // should not affect the model

	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer test-key, got %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL), WithAPIKey("test-key"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dynamic-token" {
			t.Errorf("expected Bearer dynamic-token, got %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL), WithTokenSource(provider.StaticToken("dynamic-token")))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithBaseURL_Invalid(t *testing.T) {
	o := options{baseURL: "original"}
	WithBaseURL("not-a-url")(&o)
	if o.baseURL != "original" {
		t.Errorf("expected original, got %q", o.baseURL)
	}
	// ftp scheme should be rejected
	WithBaseURL("ftp://example.com")(&o)
	if o.baseURL != "original" {
		t.Errorf("expected original after ftp://, got %q", o.baseURL)
	}
}

func TestChat_WithBaseURL_Valid(t *testing.T) {
	o := options{baseURL: "original"}
	WithBaseURL("http://custom:8080")(&o)
	if o.baseURL != "http://custom:8080" {
		t.Errorf("expected http://custom:8080, got %q", o.baseURL)
	}
}

func TestChat_WithHTTPClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	c := &http.Client{}
	model := Chat("m", WithBaseURL(server.URL), WithHTTPClient(c))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":{"message":"server error"}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m")
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("llama3")
	if model.ModelID() != "llama3" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

// --- Embedding Tests ---

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected /embeddings, got %s", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization, got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1, 0.2}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 3, "total_tokens": 3},
		})
	}))
	defer srv.Close()

	model := Embedding("nomic-embed-text", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 0.1 {
		t.Errorf("unexpected embedding: %v", result.Embeddings[0])
	}
}

func TestEmbedding_DefaultBaseURL(t *testing.T) {
	model := Embedding("nomic-embed-text")
	if model.ModelID() != "nomic-embed-text" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("m")
	if got := model.MaxValuesPerCall(); got != 2048 {
		t.Errorf("MaxValuesPerCall = %d, want 2048", got)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("nomic-embed-text")
	if model.ModelID() != "nomic-embed-text" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}
