package openai

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestImage_Generate(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("fake-png-data"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Fatalf("images = %d", len(result.Images))
	}
	if string(result.Images[0].Data) != "fake-png-data" {
		t.Errorf("data = %q", result.Images[0].Data)
	}
}

func TestImage_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_NoTokenSource(t *testing.T) {
	model := Image("dall-e-3")
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_WithSize(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["size"] != "1024x1024" {
			t.Errorf("size = %v", req["size"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1, Size: "1024x1024"})
	if err != nil {
		t.Fatal(err)
	}
}

func TestImage_ConnectionError(t *testing.T) {
	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `not json`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_InvalidBase64(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"b64_json":"!!!not-base64!!!"}]}`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_ReadBodyError(t *testing.T) {
	// Custom transport that returns a 200 OK with a body that errors on Read.
	transport := &errorBodyTransport{}
	client := &http.Client{Transport: transport}
	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL("http://fake"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

// errorBodyTransport returns a 200 response with a body that fails on Read.
type errorBodyTransport struct{}

func (t *errorBodyTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(&failReader{}),
		Header:     make(http.Header),
	}, nil
}

type failReader struct{}

func (f *failReader) Read(_ []byte) (int, error) {
	return 0, fmt.Errorf("read error")
}

func TestImage_WithHeaders(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Errorf("X-Custom = %q", r.Header.Get("X-Custom"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err != nil {
		t.Fatal(err)
	}
}

// A caller-supplied header named "Authorization" must NOT override the real
// credential: auth is applied last, so it wins.
func TestImage_AuthHeaderCannotBeOverridden(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q (caller header must not win)", got, "Bearer test-key")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("test-key"), WithBaseURL(server.URL),
		WithHeaders(map[string]string{"Authorization": "Bearer attacker"}))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1}); err != nil {
		t.Fatal(err)
	}
}

func TestImage_WithHTTPClient(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	customClient := &http.Client{}
	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL), WithHTTPClient(customClient))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Errorf("images = %d", len(result.Images))
	}
}

func TestImage_ModelID(t *testing.T) {
	model := Image("dall-e-3", WithAPIKey("k"))
	if model.ModelID() != "dall-e-3" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestImage_EnvVarResolution(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	m := Image("dall-e-3")
	im := m.(*imageModel)
	if im.opts.tokenSource == nil {
		t.Error("tokenSource should be set from OPENAI_API_KEY")
	}
}

func TestImage_EnvVarBaseURL(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
	m := Image("dall-e-3")
	im := m.(*imageModel)
	if im.opts.baseURL != "https://custom.openai.com/v1" {
		t.Errorf("baseURL = %q", im.opts.baseURL)
	}
}

func TestImage_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "https://env.url")
	m := Image("dall-e-3", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	im := m.(*imageModel)
	if im.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", im.opts.baseURL)
	}
}

// --- Item 1: gpt-image-2 must NOT receive a response_format field ---

func TestHasDefaultResponseFormat_GPTImage2(t *testing.T) {
	for _, model := range []string{"gpt-image-2", "gpt-image-2-2026-04-21"} {
		if !hasDefaultResponseFormat(model) {
			t.Errorf("hasDefaultResponseFormat(%q) = false, want true", model)
		}
	}
}

// Request direction: full golden body for gpt-image-2 — no response_format.
func TestImage_GPTImage2_GoldenRequest(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertImageRequestBody(t, r, map[string]any{
			"model":  "gpt-image-2",
			"prompt": "test",
			"n":      float64(1),
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("gpt-image-2", WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1}); err != nil {
		t.Fatal(err)
	}
}

// Response direction: gpt-image-2 b64_json fixture parses correctly.
func TestImage_GPTImage2_ResponseParsed(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("gpt-image-2-png"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s","revised_prompt":"improved"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("gpt-image-2-2026-04-21", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Fatalf("images = %d, want 1", len(result.Images))
	}
	if string(result.Images[0].Data) != "gpt-image-2-png" {
		t.Errorf("data = %q", result.Images[0].Data)
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("media type = %q, want image/png", result.Images[0].MediaType)
	}
	if got := result.ProviderMetadata["openai"]["images"].([]map[string]any)[0]["revisedPrompt"]; got != "improved" {
		t.Errorf("revisedPrompt = %v", got)
	}
}

// --- Item 2: zero-value n must not be sent ---

// Request direction: golden body confirms n is absent when zero-valued.
func TestImage_ZeroN_NotSent(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertImageRequestBody(t, r, map[string]any{
			"model":           "dall-e-3",
			"prompt":          "test",
			"response_format": "b64_json",
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test"}); err != nil {
		t.Fatal(err)
	}
}

// Request direction: golden body confirms n is sent only when N>0.
func TestImage_PositiveN_Sent(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertImageRequestBody(t, r, map[string]any{
			"model":           "dall-e-3",
			"prompt":          "test",
			"n":               float64(3),
			"response_format": "b64_json",
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 3}); err != nil {
		t.Fatal(err)
	}
}

// --- Item 3: typed output_format option + detectMediaType restricted ---

// Request direction: golden body confirms output_format is sent via the typed
// option (gpt-image-1 defaults to b64_json, so no response_format).
func TestImage_WithImageOutputFormat(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertImageRequestBody(t, r, map[string]any{
			"model":         "gpt-image-1",
			"prompt":        "test",
			"n":             float64(1),
			"output_format": "webp",
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("gpt-image-1", WithAPIKey("k"), WithBaseURL(server.URL), WithImageOutputFormat(OutputFormatWebP))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1}); err != nil {
		t.Fatal(err)
	}
}

// Request direction: golden body confirms output_format is absent by default.
func TestImage_WithoutOutputFormat_NotSent(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("img"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertImageRequestBody(t, r, map[string]any{
			"model":           "dall-e-3",
			"prompt":          "test",
			"n":               float64(1),
			"response_format": "b64_json",
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	if _, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1}); err != nil {
		t.Fatal(err)
	}
}

// Unit direction: detectMediaType restricted to png/jpeg/webp, falling back to
// png for anything outside the enum.
func TestDetectMediaType(t *testing.T) {
	cases := []struct {
		name   string
		format string
		want   string
	}{
		{"png", "png", "image/png"},
		{"jpeg", "jpeg", "image/jpeg"},
		{"jpg alias", "jpg", "image/jpeg"},
		{"webp", "webp", "image/webp"},
		// gif is not in the official output_format enum; must fall back to png.
		{"gif unsupported", "gif", "image/png"},
		{"empty", "", "image/png"},
		{"unknown", "avif", "image/png"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := detectMediaType(tc.format, ""); got != tc.want {
				t.Errorf("detectMediaType(%q) = %q, want %q", tc.format, got, tc.want)
			}
		})
	}
}

// Response direction for Item 3: the output_format echoed by the server drives
// the parsed ImageData.MediaType, restricted to png/jpeg/webp.
func TestImage_ResponseOutputFormat_MediaType(t *testing.T) {
	cases := []struct {
		name         string
		outputFormat string // emitted as top-level "output_format" in the fixture
		wantMedia    string
	}{
		{"png", "png", "image/png"},
		{"jpeg", "jpeg", "image/jpeg"},
		{"webp", "webp", "image/webp"},
		{"absent defaults to png", "", "image/png"},
		{"unsupported falls back to png", "gif", "image/png"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := base64.StdEncoding.EncodeToString([]byte("img-" + tc.name))
			fixture := fmt.Sprintf(`{"data":[{"b64_json":"%s"}]}`, encoded)
			if tc.outputFormat != "" {
				fixture = fmt.Sprintf(`{"data":[{"b64_json":"%s"}],"output_format":%q}`, encoded, tc.outputFormat)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, fixture)
			}))
			defer server.Close()

			model := Image("gpt-image-1", WithAPIKey("k"), WithBaseURL(server.URL))
			result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Images) != 1 {
				t.Fatalf("images = %d, want 1", len(result.Images))
			}
			if result.Images[0].MediaType != tc.wantMedia {
				t.Errorf("MediaType = %q, want %q", result.Images[0].MediaType, tc.wantMedia)
			}
		})
	}
}

// TestImage_ResponseBodyOverCap verifies the success-path bounded read: a
// response body larger than maxImageResponseBytes is rejected instead of being
// read fully into memory.
func TestImage_ResponseBodyOverCap(t *testing.T) {
	transport := &fixedBodyTransport{body: io.NopCloser(io.LimitReader(zeroReader{}, maxImageResponseBytes+2))}
	client := &http.Client{Transport: transport}
	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL("http://fake"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
	if err == nil {
		t.Fatal("expected error for oversized response body")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want an 'exceeds' over-cap error", err)
	}
}

// TestImage_ErrorBodyBounded verifies the error-path bounded read: an error
// response body larger than maxImageErrorBytes is truncated, so the tail marker
// never reaches the extracted error message.
func TestImage_ErrorBodyBounded(t *testing.T) {
	const tailMarker = "TAIL-MARKER-SHOULD-NOT-APPEAR"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.Copy(w, io.MultiReader(
			strings.NewReader(`{"error":{"message":"`),
			io.LimitReader(zeroReader{}, int64(maxImageErrorBytes+len(tailMarker))),
			strings.NewReader(tailMarker),
		))
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "test", N: 1})
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

// zeroReader yields an endless stream of zero bytes without allocating.
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

// assertImageRequestBody asserts the outgoing /images/generations request body
// matches want exactly (golden full-body check).
func assertImageRequestBody(t *testing.T, r *http.Request, want map[string]any) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("reading request body: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decoding request body %q: %v", body, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("request body mismatch\n got: %v\nwant: %v", got, want)
	}
}
