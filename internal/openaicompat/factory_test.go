package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func okJSONResponse() *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)),
	}
}

func testParams() provider.GenerateParams {
	return provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
}

// --- Chat factory ---

func TestNewChatModel_DoGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := NewChatModel(ChatModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenSource:   provider.StaticToken("key"),
		TokenRequired: true,
	})
	if m.ModelID() != "m" {
		t.Errorf("ModelID = %q", m.ModelID())
	}
	result, err := m.DoGenerate(t.Context(), testParams())
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q", result.Text)
	}
}

func TestNewChatModel_DoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m := NewChatModel(ChatModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenSource:   provider.StaticToken("key"),
		TokenRequired: true,
		RequestConfig: RequestConfig{IncludeStreamOptions: true},
	})
	result, err := m.DoStream(t.Context(), testParams())
	if err != nil {
		t.Fatal(err)
	}
	var got string
	for ch := range result.Stream {
		if ch.Type == provider.ChunkText {
			got += ch.Text
		}
	}
	if got != "Hi" {
		t.Errorf("stream text = %q", got)
	}
}

func TestNewChatModel_Capabilities(t *testing.T) {
	caps := provider.ModelCapabilities{Temperature: true, ToolCall: true, Reasoning: true}
	m := NewChatModel(ChatModelConfig{ModelID: "m", Capabilities: caps, TokenSource: provider.StaticToken("k")})
	got := provider.ModelCapabilitiesOf(m)
	if !got.Reasoning || !got.Temperature || !got.ToolCall {
		t.Errorf("caps = %+v", got)
	}
}

func TestNewChatModel_TokenRequiredMissing(t *testing.T) {
	m := NewChatModel(ChatModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       "http://example.invalid",
		TokenRequired: true,
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err == nil {
		t.Fatal("expected error for missing token")
	}
	if !strings.Contains(err.Error(), "no API key") {
		t.Errorf("unexpected error: %v", err)
	}

	_, err = m.DoStream(t.Context(), testParams())
	if err == nil {
		t.Fatal("expected error for missing token")
	}
}

func TestNewChatModel_TokenOptionalMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Errorf("unexpected auth header: %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := NewChatModel(ChatModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenRequired: false,
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err != nil {
		t.Fatal(err)
	}
}

func TestNewChatModel_BaseURLRequired(t *testing.T) {
	m := NewChatModel(ChatModelConfig{
		ProviderID:      "test",
		ModelID:         "m",
		BaseURLRequired: true,
		TokenSource:     provider.StaticToken("k"),
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err == nil || !strings.Contains(err.Error(), "base URL") {
		t.Fatalf("expected base URL error, got %v", err)
	}

	_, err = m.DoStream(t.Context(), testParams())
	if err == nil || !strings.Contains(err.Error(), "base URL") {
		t.Fatalf("expected base URL error, got %v", err)
	}
}

func TestNewChatModel_ExtraBody(t *testing.T) {
	var gotBody []byte
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return okJSONResponse(), nil
	})
	m := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		HTTPClient:  &http.Client{Transport: tr},
		RequestConfig: RequestConfig{
			ExtraBody: map[string]any{"custom_field": "present"},
		},
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(gotBody), "custom_field") {
		t.Errorf("request body missing extra field: %s", gotBody)
	}
}

// TestNewChatModel_RequestConfigKnobsFlowThrough proves the ARCH-1 collapse:
// ChatModelConfig embeds RequestConfig and DoGenerate / DoStream hand it
// straight to BuildRequest, so a wire-format knob set on the ChatModelConfig
// literal reaches the request body with no manual field-by-field copy. If the
// embed/copy regresses (a knob silently dropped), the wire body falls back to
// the defaults and this test fails.
func TestNewChatModel_RequestConfigKnobsFlowThrough(t *testing.T) {
	// DoGenerate: SeedKey is a RequestConfig-only knob. If the collapse drops
	// it, the request uses the default "seed" key instead of "random_seed".
	var genBody []byte
	genTR := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		genBody, _ = io.ReadAll(req.Body)
		return okJSONResponse(), nil
	})
	gen := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		HTTPClient:  &http.Client{Transport: genTR},
		RequestConfig: RequestConfig{
			SeedKey: "random_seed",
		},
	})
	seed := 7
	if _, err := gen.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Seed:     &seed,
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(genBody), `"random_seed":7`) {
		t.Errorf("SeedKey not honored via embedded RequestConfig: %s", genBody)
	}
	if strings.Contains(string(genBody), `"seed":7`) {
		t.Errorf("default seed key leaked (RequestConfig knob dropped): %s", genBody)
	}

	// DoStream: IncludeStreamOptions is a RequestConfig-only knob. If dropped,
	// stream_options.include_usage is absent from the request body.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "include_usage") {
			t.Errorf("stream_options.include_usage missing (RequestConfig knob dropped): %s", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	stream := NewChatModel(ChatModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenSource:   provider.StaticToken("k"),
		TokenRequired: true,
		RequestConfig: RequestConfig{IncludeStreamOptions: true},
	})
	if _, err := stream.DoStream(t.Context(), testParams()); err != nil {
		t.Fatal(err)
	}
}

func TestNewChatModel_Headers(t *testing.T) {
	var got string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		got = req.Header.Get("X-Test")
		return okJSONResponse(), nil
	})
	m := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		Headers:     map[string]string{"X-Test": "header-value"},
		HTTPClient:  &http.Client{Transport: tr},
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err != nil {
		t.Fatal(err)
	}
	if got != "header-value" {
		t.Errorf("X-Test = %q", got)
	}
}

func TestNewChatModel_PromptCachingWarning(t *testing.T) {
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return okJSONResponse(), nil
	})
	m := NewChatModel(ChatModelConfig{
		ProviderID:        "test-prov",
		ModelID:           "m",
		BaseURL:           "http://example.invalid",
		TokenSource:       provider.StaticToken("k"),
		HTTPClient:        &http.Client{Transport: tr},
		WarnPromptCaching: true,
	})
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	// DoStream warning branch
	streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer streamServer.Close()
	ms := NewChatModel(ChatModelConfig{
		ProviderID:        "test-prov",
		ModelID:           "m",
		BaseURL:           streamServer.URL,
		TokenSource:       provider.StaticToken("k"),
		WarnPromptCaching: true,
	})
	_, err = ms.DoStream(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatal(err)
	}
}

type failingTokenSource struct{}

func (failingTokenSource) Token(ctx context.Context) (string, error) {
	return "", errors.New("token fetch failed")
}

func TestNewChatModel_TokenSourceError(t *testing.T) {
	m := NewChatModel(ChatModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: failingTokenSource{},
	})
	_, err := m.DoGenerate(t.Context(), testParams())
	if err == nil || !strings.Contains(err.Error(), "resolving auth token") {
		t.Fatalf("expected token wrap, got %v", err)
	}
}

// --- Embedding factory ---

func TestNewEmbeddingModel_DoEmbed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1,0.2],"index":0}],"usage":{"prompt_tokens":2,"total_tokens":2}}`)
	}))
	defer server.Close()

	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenSource:   provider.StaticToken("k"),
		TokenRequired: true,
	})
	if m.ModelID() != "m" {
		t.Errorf("ModelID = %q", m.ModelID())
	}
	if m.MaxValuesPerCall() != 2048 {
		t.Errorf("MaxValuesPerCall default = %d", m.MaxValuesPerCall())
	}
	result, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || result.Embeddings[0][0] != 0.1 {
		t.Errorf("embeddings = %v", result.Embeddings)
	}
	if result.Usage.InputTokens != 2 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
}

func TestNewEmbeddingModel_MaxValuesOverride(t *testing.T) {
	m := NewEmbeddingModel(EmbeddingModelConfig{ModelID: "m", MaxValuesPerCall: 100})
	if m.MaxValuesPerCall() != 100 {
		t.Errorf("MaxValuesPerCall = %d", m.MaxValuesPerCall())
	}
}

func TestNewEmbeddingModel_BaseURLRequired(t *testing.T) {
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:      "test",
		ModelID:         "m",
		BaseURLRequired: true,
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "base URL") {
		t.Fatalf("expected base URL error, got %v", err)
	}
}

func TestNewEmbeddingModel_EncodingFormat(t *testing.T) {
	// "-" means omit encoding_format
	var gotBody []byte
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)),
		}, nil
	})
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:     "test",
		ModelID:        "m",
		BaseURL:        "http://example.invalid",
		TokenSource:    provider.StaticToken("k"),
		HTTPClient:     &http.Client{Transport: tr},
		EncodingFormat: "-",
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(gotBody), "encoding_format") {
		t.Errorf("should omit encoding_format: %s", gotBody)
	}
}

func TestNewEmbeddingModel_ProviderOptionsKeys(t *testing.T) {
	var gotBody []byte
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)),
		}, nil
	})
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:          "test",
		ModelID:             "m",
		BaseURL:             "http://example.invalid",
		TokenSource:         provider.StaticToken("k"),
		HTTPClient:          &http.Client{Transport: tr},
		ProviderOptionsKeys: []string{"dimensions"},
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 512, "other": "ignored"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(gotBody), "dimensions") {
		t.Errorf("dimensions missing: %s", gotBody)
	}
	if strings.Contains(string(gotBody), "other") {
		t.Errorf("unknown key should not be forwarded: %s", gotBody)
	}
}

func TestNewEmbeddingModel_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":"boom"}`)
	}))
	defer server.Close()
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     server.URL,
		TokenSource: provider.StaticToken("k"),
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNewEmbeddingModel_TokenSourceError(t *testing.T) {
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: failingTokenSource{},
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "resolving auth token") {
		t.Fatalf("expected token wrap, got %v", err)
	}
}

func TestNewEmbeddingModel_ConnectionError(t *testing.T) {
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://127.0.0.1:1",
		TokenSource: provider.StaticToken("k"),
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNewEmbeddingModel_OptionalToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Errorf("unexpected auth: %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
	}))
	defer server.Close()

	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:    "test",
		ModelID:       "m",
		BaseURL:       server.URL,
		TokenRequired: false,
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestNewEmbeddingModel_Headers(t *testing.T) {
	var got string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		got = req.Header.Get("X-Test")
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)),
		}, nil
	})
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		Headers:     map[string]string{"X-Test": "v"},
		HTTPClient:  &http.Client{Transport: tr},
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if got != "v" {
		t.Errorf("X-Test = %q", got)
	}
}

// TestNewEmbeddingModel_AuthCannotBeOverridden guards SEC-1: the embed path
// must apply the configured credential AFTER the merged cfg.Headers, so a
// caller-supplied Authorization header (via WithHeaders) can never strip or
// replace the provider token. Mirrors httpc.DoJSONRequest's auth-last ordering.
func TestNewEmbeddingModel_AuthCannotBeOverridden(t *testing.T) {
	var gotAuth string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)),
		}, nil
	})
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("real-key"),
		Headers:     map[string]string{"Authorization": "Bearer attacker"},
		HTTPClient:  &http.Client{Transport: tr},
	})
	if _, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{}); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-key" {
		t.Errorf("Authorization = %q, want configured 'Bearer real-key' (credential must win over cfg.Headers)", gotAuth)
	}
}

type errReader struct{}

func (errReader) Read(p []byte) (int, error) { return 0, errors.New("read boom") }
func (errReader) Close() error               { return nil }

// TestReadResponseBody_MaxInt64NoOverflow guards against the sentinel-byte
// overflow: when maxResponseBodyBytes is math.MaxInt64, the old
// maxResponseBodyBytes+1 wrapped negative and LimitReader treated it as EOF,
// silently returning an empty body. The clamped limit must read the body intact.
func TestReadResponseBody_MaxInt64NoOverflow(t *testing.T) {
	orig := maxResponseBodyBytes
	maxResponseBodyBytes = math.MaxInt64
	defer func() { maxResponseBodyBytes = orig }()

	data, err := ReadResponseBody(strings.NewReader("small"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "small" {
		t.Errorf("data = %q, want %q (overflow must not yield an empty body)", data, "small")
	}
}

func TestNewChatModel_ReadError(t *testing.T) {
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       errReader{},
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
	if err == nil || !strings.Contains(err.Error(), "reading response") {
		t.Fatalf("expected read error, got %v", err)
	}
}

func TestNewEmbeddingModel_ReadError(t *testing.T) {
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       errReader{},
		}, nil
	})
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     "http://example.invalid",
		TokenSource: provider.StaticToken("k"),
		HTTPClient:  &http.Client{Transport: tr},
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "reading response") {
		t.Fatalf("expected read error, got %v", err)
	}
}

func TestNewEmbeddingModel_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{not-json`)
	}))
	defer server.Close()
	m := NewEmbeddingModel(EmbeddingModelConfig{
		ProviderID:  "test",
		ModelID:     "m",
		BaseURL:     server.URL,
		TokenSource: provider.StaticToken("k"),
	})
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "parsing response") {
		t.Fatalf("expected parsing error, got %v", err)
	}
}
