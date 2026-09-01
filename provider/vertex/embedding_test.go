package vertex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/models/text-embedding-004:predict") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		if len(instances) != 1 {
			t.Fatalf("expected 1 instance, got %d", len(instances))
		}
		inst := instances[0].(map[string]any)
		if inst["content"] != "hello" {
			t.Errorf("content = %v", inst["content"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{
					"embeddings": map[string]any{
						"values":     []float64{0.1, 0.2, 0.3},
						"statistics": map[string]any{"token_count": 1},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(result.Embeddings[0]))
	}
	if result.Usage.InputTokens != 1 {
		t.Errorf("expected 1 input token, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_MultipleValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		if len(instances) != 3 {
			t.Errorf("expected 3 instances, got %d", len(instances))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1, 0.2}, "statistics": map[string]any{"token_count": 1}}},
				{"embeddings": map[string]any{"values": []float64{0.3, 0.4}, "statistics": map[string]any{"token_count": 2}}},
				{"embeddings": map[string]any{"values": []float64{0.5, 0.6}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b", "c"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Embeddings))
	}
	if result.Usage.InputTokens != 4 {
		t.Errorf("expected 4 total input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_ProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Check instances have task_type and title.
		instances := req["instances"].([]any)
		inst := instances[0].(map[string]any)
		if inst["task_type"] != "RETRIEVAL_QUERY" {
			t.Errorf("task_type = %v", inst["task_type"])
		}
		if inst["title"] != "My Doc" {
			t.Errorf("title = %v", inst["title"])
		}

		// Check parameters have outputDimensionality and autoTruncate.
		params := req["parameters"].(map[string]any)
		if params["outputDimensionality"] != float64(256) {
			t.Errorf("outputDimensionality = %v", params["outputDimensionality"])
		}
		if params["autoTruncate"] != true {
			t.Errorf("autoTruncate = %v", params["autoTruncate"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"taskType":             "RETRIEVAL_QUERY",
				"title":                "My Doc",
				"outputDimensionality": 256,
				"autoTruncate":         true,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedding_NoProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		inst := instances[0].(map[string]any)
		if _, ok := inst["task_type"]; ok {
			t.Error("task_type should not be set")
		}
		if _, ok := req["parameters"]; ok {
			t.Error("parameters should not be set when empty")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request","code":400}}`))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_NoProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "PROJECT required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	if model.ModelID() != "text-embedding-004" {
		t.Errorf("ModelID = %q", model.ModelID())
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("text-embedding-004", WithTokenSource(provider.StaticToken("tok")))
	if got := model.MaxValuesPerCall(); got != 250 {
		t.Errorf("MaxValuesPerCall = %d, want 250", got)
	}
}

func TestEmbedding_ConnectionError(t *testing.T) {
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://127.0.0.1:1"),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_UnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

// TestEmbedding_GeminiUnmarshalError covers the error branch of the Gemini
// batchEmbedContents response parse (embedding.go:169-171): with API-key auth
// the server returns malformed JSON, so json.Unmarshal into the geminiResult
// struct must fail with a "parsing response" error.
func TestEmbedding_GeminiUnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, ":batchEmbedContents") {
			t.Errorf("path = %s, want ...:batchEmbedContents", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings": not-valid-json`)
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_TokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token failed")
	})
	model := Embedding("text-embedding-004", WithTokenSource(ts), WithBaseURL("http://fake"))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Custom": "val"}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if gotHeader != "val" {
		t.Errorf("X-Custom = %q", gotHeader)
	}
}

func TestEmbedding_AuthHeaderWinsOverWithHeaders(t *testing.T) {
	// A caller-supplied WithHeaders header named like the auth header must
	// NOT override the real credential on the OAuth :predict path.
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("real-token")),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"Authorization": "Bearer spoofed-token"}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-token" {
		t.Errorf("Authorization = %q, want %q (credential must win over WithHeaders)", gotAuth, "Bearer real-token")
	}
}

func TestEmbedding_DefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-east1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	expected := "https://us-east1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-east1/publishers/google/models/text-embedding-004:predict"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestEmbedding_NoTokenSource(t *testing.T) {
	// No token source + custom baseURL -- auth is skipped, request sent unauthenticated.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("unexpected auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("got %d embeddings, want 1", len(result.Embeddings))
	}
}

func TestEmbedding_ReadBodyError(t *testing.T) {
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
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

// embedAPIKeyURLTransport captures the URL and request body for API-key
// embedding requests. It responds with the Gemini batchEmbedContents shape.
type embedAPIKeyURLTransport struct {
	captured string
	body     map[string]any
}

func (tr *embedAPIKeyURLTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.captured = req.URL.String()
	if req.Body != nil {
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &tr.body)
	}
	body := `{"embeddings":[{"value":[0.1]}]}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}, nil
}

func TestEmbedding_APIKeySkipsBearerAuth(t *testing.T) {
	transport := &embedAPIKeyURLTransport{}
	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	// URL should contain ?key=
	if !strings.Contains(transport.captured, "?key=my-embed-key") {
		t.Errorf("URL should contain ?key=, got %q", transport.captured)
	}
	// Should use Gemini API base URL.
	if !strings.Contains(transport.captured, "generativelanguage.googleapis.com") {
		t.Errorf("URL should use generativelanguage.googleapis.com, got %q", transport.captured)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

// TestEmbedding_APIKeyUsesGeminiBodyShape verifies that API-key auth emits the
// Gemini batchEmbedContents body shape (item #27) instead of the Vertex
// :predict instances/parameters shape.
func TestEmbedding_APIKeyUsesGeminiBodyShape(t *testing.T) {
	transport := &embedAPIKeyURLTransport{}
	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello world"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"taskType":             "RETRIEVAL_QUERY",
				"title":                "My Doc",
				"outputDimensionality": 256,
				"autoTruncate":         true,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Endpoint must be batchEmbedContents, not :predict.
	if !strings.Contains(transport.captured, ":batchEmbedContents") {
		t.Errorf("expected :batchEmbedContents in URL, got %q", transport.captured)
	}

	// Body must use the Gemini "requests" shape, not Vertex "instances".
	if transport.body == nil {
		t.Fatal("no request body captured")
	}
	if _, ok := transport.body["instances"]; ok {
		t.Error("Gemini body must NOT contain Vertex 'instances'")
	}
	if _, ok := transport.body["parameters"]; ok {
		t.Error("Gemini body must NOT contain Vertex 'parameters'")
	}
	requests, ok := transport.body["requests"].([]any)
	if !ok || len(requests) != 1 {
		t.Fatalf("expected 1 Gemini request, got %#v", transport.body["requests"])
	}
	req := requests[0].(map[string]any)
	if req["model"] != "models/text-embedding-004" {
		t.Errorf("request model = %v", req["model"])
	}
	content, ok := req["content"].(map[string]any)
	if !ok {
		t.Fatalf("request content = %#v", req["content"])
	}
	parts := content["parts"].([]any)
	part := parts[0].(map[string]any)
	if part["text"] != "hello world" {
		t.Errorf("part text = %v", part["text"])
	}
	// Gemini request-level fields (autoTruncate is Vertex-only and dropped).
	if req["taskType"] != "RETRIEVAL_QUERY" {
		t.Errorf("taskType = %v", req["taskType"])
	}
	if req["title"] != "My Doc" {
		t.Errorf("title = %v", req["title"])
	}
	if req["outputDimensionality"] != float64(256) {
		t.Errorf("outputDimensionality = %v", req["outputDimensionality"])
	}
	if _, ok := req["autoTruncate"]; ok {
		t.Error("autoTruncate must not be sent to the Gemini API")
	}
}

// TestEmbedding_APIKeyMultipleValues verifies the Gemini batch body carries
// one request per value and parses the embeddings array.
func TestEmbedding_APIKeyMultipleValues(t *testing.T) {
	transport := &embedAPIKeyURLTransport{}
	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"a", "b", "c"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	requests, ok := transport.body["requests"].([]any)
	if !ok || len(requests) != 3 {
		t.Fatalf("expected 3 Gemini requests, got %#v", transport.body["requests"])
	}
}

// TestEmbedding_APIKeyGeminiGoldenBody is a golden full-body REQUEST contract
// test (item #27): API-key auth must emit the exact Gemini
// batchEmbedContents body shape -- {"requests":[{model,content:{parts:[{text}]},
// taskType,title,outputDimensionality}]} -- with no Vertex :predict keys
// (instances/parameters) and no Vertex-only fields (autoTruncate).
func TestEmbedding_APIKeyGeminiGoldenBody(t *testing.T) {
	transport := &embedAPIKeyURLTransport{}
	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello world"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"taskType":             "RETRIEVAL_QUERY",
				"title":                "My Doc",
				"outputDimensionality": 256,
				"autoTruncate":         true,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]any{
		"requests": []any{
			map[string]any{
				"model":                "models/text-embedding-004",
				"content":              map[string]any{"parts": []any{map[string]any{"text": "hello world"}}},
				"taskType":             "RETRIEVAL_QUERY",
				"title":                "My Doc",
				"outputDimensionality": float64(256),
			},
		},
	}
	if !reflect.DeepEqual(transport.body, want) {
		gotJSON, _ := json.Marshal(transport.body)
		wantJSON, _ := json.Marshal(want)
		t.Errorf("request body mismatch:\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}

// TestEmbedding_APIKeyParsesGeminiResponse is a RESPONSE-direction contract
// test (item #27): a Gemini batchEmbedContents response
// {"embeddings":[{"value":[...]}]} must be parsed into the EmbedResult.
func TestEmbedding_APIKeyParsesGeminiResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, ":batchEmbedContents") {
			t.Errorf("path = %s, want ...:batchEmbedContents", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "key=my-embed-key") {
			t.Errorf("query = %q, want key=my-embed-key", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":[{"value":[0.1,0.2,0.3]},{"value":[0.4,0.5,0.6]}]}`)
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithAPIKey("my-embed-key"),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
	wantFirst := []float64{0.1, 0.2, 0.3}
	wantSecond := []float64{0.4, 0.5, 0.6}
	if !reflect.DeepEqual(result.Embeddings[0], wantFirst) {
		t.Errorf("embedding[0] = %v, want %v", result.Embeddings[0], wantFirst)
	}
	if !reflect.DeepEqual(result.Embeddings[1], wantSecond) {
		t.Errorf("embedding[1] = %v, want %v", result.Embeddings[1], wantSecond)
	}
	if result.Response.Model != "text-embedding-004" {
		t.Errorf("Response.Model = %q, want text-embedding-004", result.Response.Model)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"embeddings": map[string]any{"values": []float64{0.1}, "statistics": map[string]any{"token_count": 1}}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHTTPClient(&http.Client{}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_ResponseModelPopulated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{
					"embeddings": map[string]any{
						"values":     []float64{0.1, 0.2},
						"statistics": map[string]any{"token_count": 1},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Response.Model != "text-embedding-004" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "text-embedding-004")
	}
}

func TestEmbedding_ResponseBodyTooLarge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Body larger than maxVertexSuccessBodyBytes (64 MiB) must be rejected.
		_, _ = w.Write(make([]byte, maxVertexSuccessBodyBytes+1))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for oversized success response")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want it to mention the size limit", err.Error())
	}
}

func TestEmbedding_ErrorBodyTooLarge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		// Error body larger than maxVertexErrorBodyBytes (1 MiB) must be bounded.
		_, _ = w.Write(make([]byte, maxVertexErrorBodyBytes+1))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error for HTTP 500 response")
	}
}
