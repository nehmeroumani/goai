package vertex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

type requestTrackingTransport struct {
	responseBody     io.ReadCloser
	requestCollected chan struct{}
}

func (t *requestTrackingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	runtime.AddCleanup(req, func(collected chan struct{}) {
		close(collected)
	}, t.requestCollected)

	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       t.responseBody,
		Request:    req,
	}, nil
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/chat/completions") {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(server.URL))
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

func TestChat_Stream_DoesNotRetainRequestGraph(t *testing.T) {
	reader, writer := io.Pipe()
	t.Cleanup(func() {
		_ = reader.Close()
		_ = writer.Close()
	})

	requestCollected := make(chan struct{})
	client := &http.Client{Transport: &requestTrackingTransport{
		responseBody:     reader,
		requestCollected: requestCollected,
	}}
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL("http://vertex.test"),
		WithHTTPClient(client))

	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// The pipe keeps the stream active. The request and its serialized body
	// should nevertheless be unreachable once DoStream returns.
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		runtime.GC()
		select {
		case <-requestCollected:
			cancel()
			for {
				select {
				case _, ok := <-result.Stream:
					if !ok {
						return
					}
				case <-timer.C:
					t.Fatal("stream did not close after context cancellation")
				}
			}
		case <-timer.C:
			t.Fatal("active stream retained its response/request graph")
		case <-ticker.C:
		}
	}
}

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-123","model":"gemini-2.5-pro","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
		t.Errorf("Text = %q", result.Text)
	}
}

func TestChat_NativeGeminiVertex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer oauth-token" {
			t.Errorf("Authorization = %q, want Bearer oauth-token", got)
		}
		if got := r.Header.Get("X-Custom"); got != "value" {
			t.Errorf("X-Custom = %q, want value", got)
		}
		if !strings.HasSuffix(r.URL.Path, "/gemini-2.5-pro:generateContent") {
			t.Errorf("path = %q, want native generateContent endpoint", r.URL.Path)
		}
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"native"}]},"finishReason":"STOP"}]}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro",
		WithNativeGemini(),
		WithProject("my-proj"),
		WithLocation("global"),
		WithTokenSource(provider.StaticToken("oauth-token")),
		WithNativeChatBaseURL(server.URL+"/models"),
		WithHeaders(map[string]string{"X-Custom": "value"}),
		WithHTTPClient(server.Client()))

	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "native" {
		t.Errorf("Text = %q, want native", result.Text)
	}
	if _, ok := model.(provider.FileUploadCapableModel); ok {
		t.Fatalf("native Vertex model type %T must not implement FileUploadCapableModel", model)
	}
}

func TestChat_NativeGeminiRejectsAPIKey(t *testing.T) {
	model := Chat("gemini-2.5-flash",
		WithNativeGemini(),
		WithAPIKey("api-key"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "OAuth token source") {
		t.Fatalf("DoGenerate() error = %v, want OAuth token source error", err)
	}
}

func TestChat_NativeGeminiRejectsCompatBaseURL(t *testing.T) {
	model := Chat("gemini-2.5-pro",
		WithNativeGemini(),
		WithTokenSource(provider.StaticToken("oauth-token")),
		WithBaseURL("https://compat.example.test"))
	if got := model.ModelID(); got != "gemini-2.5-pro" {
		t.Errorf("ModelID() = %q, want gemini-2.5-pro", got)
	}
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
	want := "use WithNativeChatBaseURL"
	if _, err := model.DoGenerate(t.Context(), params); err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("DoGenerate() error = %v, want %q", err, want)
	}
	if _, err := model.DoStream(t.Context(), params); err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("DoStream() error = %v, want %q", err, want)
	}
}

func TestChat_NativeChatBaseURLRequiresNativeTransport(t *testing.T) {
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("oauth-token")),
		WithNativeChatBaseURL("https://vertex.example.test/models"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "requires WithNativeGemini") {
		t.Fatalf("DoGenerate() error = %v, want native transport error", err)
	}
}

func TestChatOnlyOptionsRejectedByOtherConstructors(t *testing.T) {
	want := "only supported by Chat"
	token := WithTokenSource(provider.StaticToken("oauth-token"))

	t.Run("image", func(t *testing.T) {
		model := Image("imagen-4.0-generate-001", token, WithNativeChatBaseURL("https://vertex.example.test/models"))
		_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "draw", N: 1})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("DoGenerate() error = %v, want %q", err, want)
		}
	})

	t.Run("embedding", func(t *testing.T) {
		model := Embedding("text-embedding-004", token, WithNativeGemini())
		_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("DoEmbed() error = %v, want %q", err, want)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		model := AnthropicChat("claude-sonnet-4", token, WithNativeChatBaseURL("https://vertex.example.test/models"))
		_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
			Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("DoGenerate() error = %v, want %q", err, want)
		}
	})
}

func TestNoProject(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	model := Chat("model", WithTokenSource(provider.StaticToken("tok")))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "GOOGLE_CLOUD_PROJECT") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("model", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDefaultLocation(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithProject("proj"))
	cm := model.(*chatModel)
	if cm.opts.location != "us-central1" {
		t.Errorf("location = %q, want us-central1", cm.opts.location)
	}
}

func TestCustomLocation(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithLocation("europe-west1"))
	cm := model.(*chatModel)
	if cm.opts.location != "europe-west1" {
		t.Errorf("location = %q", cm.opts.location)
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"))
	if model.ModelID() != "gemini-2.5-pro" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://127.0.0.1:1"))
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

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
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
		WithTokenSource(provider.StaticToken("tok")),
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

func TestTokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token fetch failed")
	})
	model := Chat("m", WithTokenSource(ts), WithBaseURL("http://fake"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestDefaultURLWithProject(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	// Use baseURL to point at test server, but verify project/location env fallbacks work.
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "my-project")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	cm := model.(*chatModel)
	if cm.opts.project != "my-project" {
		t.Errorf("project = %q, want my-project", cm.opts.project)
	}
	if cm.opts.location != "europe-west4" {
		t.Errorf("location = %q, want europe-west4", cm.opts.location)
	}
}

func TestNoTokenSource(t *testing.T) {
	// No token source + custom baseURL -- auth is skipped, request goes through unauthenticated.
	// With a fake URL this should fail with a connection error.
	model := Chat("m", WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for unreachable endpoint")
	}
	// Should be a connection error, not an auth error.
	if strings.Contains(err.Error(), "no API key") {
		t.Errorf("should not be auth error: %v", err)
	}
}

type urlCapturingTransport struct {
	captured string
}

func (tr *urlCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.captured = req.URL.String()
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)),
		Header:     make(http.Header),
	}, nil
}

func TestAuthHeaderWinsOverWithHeaders(t *testing.T) {
	// A caller-supplied WithHeaders header named like the auth header must
	// NOT override the real credential on the chat doHTTP path.
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("real-token")),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"Authorization": "Bearer spoofed-token"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-token" {
		t.Errorf("Authorization = %q, want %q (credential must win over WithHeaders)", gotAuth, "Bearer real-token")
	}
}

func TestRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-Header") != "from-params" {
			t.Errorf("X-Request-Header = %q", r.Header.Get("X-Request-Header"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Request-Header": "from-params"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestWithAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer my-api-key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("my-api-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- autoTokenSource + resolveOpts auto-resolve tests ---

func TestAutoTokenSource_GoogleAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "gkey-123")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	if ts == nil {
		t.Fatal("expected non-nil token source")
	}
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gkey-123" {
		t.Errorf("key = %q, want gkey-123", aks.key)
	}
}

func TestAutoTokenSource_GeminiAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "gemini-456")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gemini-456" {
		t.Errorf("key = %q, want gemini-456", aks.key)
	}
}

func TestAutoTokenSource_GoogleGenerativeAIAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "gen-789")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gen-789" {
		t.Errorf("key = %q, want gen-789", aks.key)
	}
}

func TestAutoTokenSource_FallsBackToADC(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	// ADC will also fail without real creds, but autoTokenSource still returns a non-nil source.
	ts := autoTokenSource(false)
	if ts == nil {
		t.Fatal("expected non-nil token source from ADC fallback")
	}
	// Should NOT be an apiKeyTokenSource.
	if _, ok := ts.(*apiKeyTokenSource); ok {
		t.Error("expected ADC-based token source, got apiKeyTokenSource")
	}
}

func TestResolveOpts_AutoResolveAuth(t *testing.T) {
	// No explicit tokenSource, no baseURL → should auto-resolve.
	t.Setenv("GOOGLE_API_KEY", "auto-key")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")

	o := resolveOpts(nil)
	if o.tokenSource == nil {
		t.Fatal("expected auto-resolved token source")
	}
	aks, ok := o.tokenSource.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource from auto-resolve")
	}
	if aks.key != "auto-key" {
		t.Errorf("key = %q, want auto-key", aks.key)
	}
}

// --- API key URL routing tests ---

func TestResolveURL_APIKeyUsesGeminiEndpoint(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("gemini-2.5-pro",
		WithAPIKey("my-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	expected := "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestWireModelID_APIKeyReturnsBareModelName(t *testing.T) {
	model := Chat("gemini-2.5-pro", WithAPIKey("my-key"))
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want gemini-2.5-pro", got)
	}
}

func TestWireModelID_VertexAddsPrefix(t *testing.T) {
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("proj"),
	)
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "google/gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want google/gemini-2.5-pro", got)
	}
}

func TestWireModelID_AlreadyPrefixed(t *testing.T) {
	model := Chat("google/gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("proj"),
	)
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "google/gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want google/gemini-2.5-pro (should not double-prefix)", got)
	}
}

func TestNativeBaseURL_APIKeyUsesGeminiAPI(t *testing.T) {
	o := options{
		tokenSource: &apiKeyTokenSource{key: "test-key"},
	}
	got := nativeBaseURL(o)
	expected := "https://generativelanguage.googleapis.com/v1beta"
	if got != expected {
		t.Errorf("nativeBaseURL() = %q, want %q", got, expected)
	}
}

func TestNativeURL_APIKeyAppendsKeyParam(t *testing.T) {
	o := options{
		tokenSource: &apiKeyTokenSource{key: "test-key"},
	}
	got, err := nativeURL(o, "models/text-embedding-004:predict")
	if err != nil {
		t.Fatal(err)
	}
	expected := "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:predict?key=test-key"
	if got != expected {
		t.Errorf("nativeURL() = %q, want %q", got, expected)
	}
}

func TestAutoTokenSource_HasProjectUsesADC(t *testing.T) {
	// When hasProject=true, autoTokenSource should prefer ADC (not API key env vars).
	t.Setenv("GOOGLE_API_KEY", "should-not-use")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	ts := autoTokenSource(true)
	if ts == nil {
		t.Fatal("expected non-nil token source")
	}
	// Should NOT be an apiKeyTokenSource -- ADC is preferred when project is set.
	if _, ok := ts.(*apiKeyTokenSource); ok {
		t.Error("expected ADC token source, got apiKeyTokenSource")
	}
}

func TestStripGeminiProviderOptions_MapsThinkingConfigToReasoningEffort(t *testing.T) {
	params := &provider.GenerateParams{
		ProviderOptions: map[string]any{
			"thinkingConfig": map[string]any{"thinkingBudget": 1024},
			"otherOption":    "keep",
		},
	}
	stripGeminiProviderOptions(params)
	if _, ok := params.ProviderOptions["thinkingConfig"]; ok {
		t.Error("expected thinkingConfig to be removed")
	}
	// Budget 1024 (< 8192) maps to medium.
	if got := params.ProviderOptions["reasoning_effort"]; got != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", got)
	}
	if _, ok := params.ProviderOptions["otherOption"]; !ok {
		t.Error("expected otherOption to be kept")
	}
}

func TestStripGeminiProviderOptions_ThinkingBudgetBuckets(t *testing.T) {
	cases := []struct {
		name   string
		tc     any
		want   string
		hasEff bool
	}{
		{"zero budget -> low", map[string]any{"thinkingBudget": 0}, "low", true},
		{"small budget -> low", map[string]any{"thinkingBudget": 64}, "low", true},
		{"medium budget -> medium", map[string]any{"thinkingBudget": 4096}, "medium", true},
		{"default budget -> high", map[string]any{"thinkingBudget": 8192}, "high", true},
		{"large budget -> high", map[string]any{"thinkingBudget": 98304}, "high", true},
		{"no budget key -> default medium", map[string]any{}, "medium", true},
		{"float budget", map[string]any{"thinkingBudget": float64(4096)}, "medium", true},
		{"json.Number budget", map[string]any{"thinkingBudget": json.Number("4096")}, "medium", true},
		{"json.Number invalid -> default", map[string]any{"thinkingBudget": json.Number("abc")}, "medium", true},
		{"string budget -> default", map[string]any{"thinkingBudget": "abc"}, "medium", true},
		{"bool true -> default medium", true, "medium", true},
		{"bool false -> disabled", false, "", false},
		{"unrecognized -> disabled", "nonsense", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := &provider.GenerateParams{
				ProviderOptions: map[string]any{"thinkingConfig": tc.tc},
			}
			stripGeminiProviderOptions(params)
			got, ok := params.ProviderOptions["reasoning_effort"]
			if ok != tc.hasEff {
				t.Fatalf("reasoning_effort present = %v, want %v", ok, tc.hasEff)
			}
			if tc.hasEff && got != tc.want {
				t.Errorf("reasoning_effort = %v, want %v", got, tc.want)
			}
			if _, stillThere := params.ProviderOptions["thinkingConfig"]; stillThere {
				t.Error("thinkingConfig should always be removed")
			}
		})
	}
}

func TestStripGeminiProviderOptions_NilOptions(t *testing.T) {
	params := &provider.GenerateParams{}
	stripGeminiProviderOptions(params) // should not panic
}

// TestChat_ThinkingConfigMapsToReasoningEffortOnWire is a golden full-body
// REQUEST contract test (item #29): a caller passing Gemini-native
// thinkingConfig in ProviderOptions must see it stripped from the outgoing
// OpenAI-compat body and translated to the reasoning_effort knob, so the
// OpenAI-compatible Vertex endpoint receives a schema it understands.
func TestChat_ThinkingConfigMapsToReasoningEffortOnWire(t *testing.T) {
	var capturedBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		capturedBody = string(raw)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"thinkingConfig": map[string]any{"thinkingBudget": 4096},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(capturedBody), &got); err != nil {
		t.Fatalf("captured body not JSON: %v (%q)", err, capturedBody)
	}
	// thinkingConfig must be gone, reasoning_effort present (4096 < 8192 → medium).
	if _, ok := got["thinkingConfig"]; ok {
		t.Error("thinkingConfig must not be sent to the OpenAI-compat endpoint")
	}
	if got["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", got["reasoning_effort"])
	}
	// Golden full-body assertion: only the OpenAI-compat keys, nothing leaked.
	want := map[string]any{
		"model":            "google/gemini-2.5-pro",
		"stream":           false,
		"messages":         []any{map[string]any{"role": "user", "content": "hi"}},
		"reasoning_effort": "medium",
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(want)
		t.Errorf("request body mismatch:\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}

// TestChat_ThinkingConfigDisabledOmitsReasoningEffort verifies that a
// thinkingConfig:false (disabled) does not emit reasoning_effort at all.
func TestChat_ThinkingConfigDisabledOmitsReasoningEffort(t *testing.T) {
	var capturedBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		capturedBody = string(raw)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"thinkingConfig": false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(capturedBody), &got); err != nil {
		t.Fatalf("captured body not JSON: %v (%q)", err, capturedBody)
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Error("reasoning_effort must not be emitted when thinking is disabled")
	}
	if _, ok := got["thinkingConfig"]; ok {
		t.Error("thinkingConfig must be stripped")
	}
}

func TestSanitizeToolSchemas_CleansTool(t *testing.T) {
	// Schema with additionalProperties (which Gemini doesn't support).
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}
	raw, _ := json.Marshal(schema)
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "test_tool", InputSchema: raw},
		},
	}
	sanitizeToolSchemas(params)
	// After sanitization, additionalProperties should be removed.
	var result map[string]any
	if err := json.Unmarshal(params.Tools[0].InputSchema, &result); err != nil {
		t.Fatal(err)
	}
	if _, ok := result["additionalProperties"]; ok {
		t.Error("expected additionalProperties to be removed by sanitization")
	}
}

func TestSanitizeToolSchemas_EmptySchema(t *testing.T) {
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "no_schema", InputSchema: nil},
		},
	}
	sanitizeToolSchemas(params) // should skip gracefully
}

func TestSanitizeToolSchemas_InvalidJSON(t *testing.T) {
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "bad_json", InputSchema: json.RawMessage(`{invalid}`)},
		},
	}
	sanitizeToolSchemas(params) // should skip gracefully, not panic
}

func TestResolveOpts_EnvVarBaseURL(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "https://custom.vertex.com")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "myproject")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "us-east1")
	o := resolveOpts(nil)
	if o.baseURL != "https://custom.vertex.com" {
		t.Errorf("baseURL = %q", o.baseURL)
	}
	if o.project != "myproject" {
		t.Errorf("project = %q", o.project)
	}
	if o.location != "us-east1" {
		t.Errorf("location = %q", o.location)
	}
}

func TestResolveOpts_FallbackEnvVars(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "cloud-proj")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "")
	o := resolveOpts(nil)
	if o.project != "cloud-proj" {
		t.Errorf("project = %q, want cloud-proj", o.project)
	}
	if o.location != "asia-east1" {
		t.Errorf("location = %q, want asia-east1", o.location)
	}
}

func TestResolveOpts_GcloudProjectFallback(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "gcloud-proj")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "")
	o := resolveOpts(nil)
	if o.project != "gcloud-proj" {
		t.Errorf("project = %q, want gcloud-proj", o.project)
	}
	// location should default to us-central1
	if o.location != "us-central1" {
		t.Errorf("location = %q, want us-central1", o.location)
	}
}

// --- Coverage: setAuth ---

func TestSetAuth_NilTokenSource(t *testing.T) {
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, nil)
	if err != nil {
		t.Fatalf("expected nil error for nil token source, got %v", err)
	}
	if req.Header.Get("Authorization") != "" {
		t.Error("Authorization header should not be set for nil token source")
	}
}

func TestSetAuth_TokenError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token error")
	})
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, ts)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("error = %v", err)
	}
}

func TestSetAuth_Success(t *testing.T) {
	ts := provider.StaticToken("my-token")
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, ts)
	if err != nil {
		t.Fatal(err)
	}
	if req.Header.Get("Authorization") != "Bearer my-token" {
		t.Errorf("Authorization = %q", req.Header.Get("Authorization"))
	}
}

func TestSanitizeToolSchemas_MarshalError(t *testing.T) {
	// Swap jsonMarshalFunc to simulate a marshal error.
	orig := jsonMarshalFunc
	jsonMarshalFunc = func(v any) ([]byte, error) {
		return nil, fmt.Errorf("forced marshal error")
	}
	defer func() { jsonMarshalFunc = orig }()

	schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "tool1", InputSchema: schema},
		},
	}
	sanitizeToolSchemas(params)
	// Schema should remain unchanged (marshal error → continue).
	if string(params.Tools[0].InputSchema) != string(schema) {
		t.Error("expected schema unchanged after marshal error")
	}
}

func TestValidGCPIdentifier(t *testing.T) {
	valid := []string{
		"us-central1",
		"my-project-123",
		"example.com:my-project", // domain-scoped project
		"a",
	}
	for _, v := range valid {
		if !validGCPIdentifier(v) {
			t.Errorf("expected %q to be valid", v)
		}
	}
	invalid := []string{
		"",
		"../evil",
		"us-central1/../../x",
		"has spaces",
		"-starts-with-dash",
		"foo..bar", // path traversal
	}
	for _, v := range invalid {
		if validGCPIdentifier(v) {
			t.Errorf("expected %q to be invalid", v)
		}
	}
}

func TestNativeURL_InvalidLocation(t *testing.T) {
	o := options{
		tokenSource: provider.StaticToken("tok"),
		project:     "my-project",
		location:    "../evil",
	}
	_, err := nativeURL(o, "models/text-embedding-004:predict")
	if err == nil {
		t.Fatal("expected error for invalid location")
	}
	if !strings.Contains(err.Error(), "invalid location") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestNativeURL_InvalidProject(t *testing.T) {
	o := options{
		tokenSource: provider.StaticToken("tok"),
		project:     "has spaces",
		location:    "us-central1",
	}
	_, err := nativeURL(o, "models/text-embedding-004:predict")
	if err == nil {
		t.Fatal("expected error for invalid project")
	}
	if !strings.Contains(err.Error(), "invalid project") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestResolveURL_InvalidLocation(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "my-project")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "../evil")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"), WithLocation("../evil"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid location")
	}
	if !strings.Contains(err.Error(), "invalid location") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestResolveURL_InvalidProject(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")),
		WithProject("has spaces"), WithLocation("us-central1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid project")
	}
	if !strings.Contains(err.Error(), "invalid project") {
		t.Errorf("unexpected error: %s", err)
	}
}

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to the Vertex
// provider succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-test","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(server.URL))

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
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer streamServer.Close()

	streamModel := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(streamServer.URL))

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
