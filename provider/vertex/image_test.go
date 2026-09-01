package vertex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestImage_Generate(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-png-data"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/models/imagen-3.0-generate-002:predict") {
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
		if inst["prompt"] != "a cat" {
			t.Errorf("prompt = %v", inst["prompt"])
		}

		params := req["parameters"].(map[string]any)
		if params["sampleCount"] != float64(2) {
			t.Errorf("sampleCount = %v", params["sampleCount"])
		}
		if params["aspectRatio"] != "16:9" {
			t.Errorf("aspectRatio = %v", params["aspectRatio"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt:      "a cat",
		N:           2,
		AspectRatio: "16:9",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Images) != 2 {
		t.Fatalf("expected 2 images, got %d", len(result.Images))
	}
	if string(result.Images[0].Data) != "fake-png-data" {
		t.Errorf("unexpected image data: %q", string(result.Images[0].Data))
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("mediaType = %q", result.Images[0].MediaType)
	}
}

func TestImage_AuthHeaderWinsOverWithHeaders(t *testing.T) {
	// A caller-supplied WithHeaders header named like the auth header must
	// NOT override the real credential on the :predict path.
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-png-data"))
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("real-token")),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"Authorization": "Bearer spoofed-token"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "a cat", N: 1})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer real-token" {
		t.Errorf("Authorization = %q, want %q (credential must win over WithHeaders)", gotAuth, "Bearer real-token")
	}
}

func TestImage_ProviderOptions(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		params := req["parameters"].(map[string]any)
		if params["negativePrompt"] != "violence" {
			t.Errorf("negativePrompt = %v", params["negativePrompt"])
		}
		if params["personGeneration"] != "allow_adult" {
			t.Errorf("personGeneration = %v", params["personGeneration"])
		}
		if params["safetySetting"] != "block_only_high" {
			t.Errorf("safetySetting = %v", params["safetySetting"])
		}
		if params["addWatermark"] != true {
			t.Errorf("addWatermark = %v", params["addWatermark"])
		}
		if params["sampleImageSize"] != "2K" {
			t.Errorf("sampleImageSize = %v", params["sampleImageSize"])
		}
		if params["seed"] != float64(42) {
			t.Errorf("seed = %v", params["seed"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"negativePrompt":   "violence",
				"personGeneration": "allow_adult",
				"safetySetting":    "block_only_high",
				"addWatermark":     true,
				"sampleImageSize":  "2K",
				"seed":             42,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImage_RevisedPrompt(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png", "prompt": "a realistic cat"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.ProviderMetadata == nil {
		t.Fatal("expected ProviderMetadata")
	}
	vertexMeta := result.ProviderMetadata["vertex"]
	images := vertexMeta["images"].([]map[string]any)
	if images[0]["revisedPrompt"] != "a realistic cat" {
		t.Errorf("revisedPrompt = %v", images[0]["revisedPrompt"])
	}
}

func TestImage_NoRevisedPrompt(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.ProviderMetadata != nil {
		t.Errorf("expected nil ProviderMetadata, got %v", result.ProviderMetadata)
	}
}

func TestImage_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request","code":400}}`))
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImage_NoProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	model := Image("imagen-3.0-generate-002", WithTokenSource(provider.StaticToken("tok")))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "PROJECT required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestImage_ModelID(t *testing.T) {
	model := Image("imagen-3.0-generate-002", WithTokenSource(provider.StaticToken("tok")))
	if model.ModelID() != "imagen-3.0-generate-002" {
		t.Errorf("ModelID = %q", model.ModelID())
	}
}

func TestImage_ConnectionError(t *testing.T) {
	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://127.0.0.1:1"),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestImage_TokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token failed")
	})
	model := Image("imagen-3.0-generate-002", WithTokenSource(ts), WithBaseURL("http://fake"))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestImage_DefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	expected := "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestImage_WithHeaders(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Custom": "val"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotHeader != "val" {
		t.Errorf("X-Custom = %q", gotHeader)
	}
}

func TestImage_NoTokenSource(t *testing.T) {
	// No token source + custom baseURL -- auth is skipped, request sent unauthenticated.
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("unexpected auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002", WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Images) != 1 {
		t.Errorf("got %d images, want 1", len(result.Images))
	}
}

func TestImage_InvalidBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": "not-valid-base64!!!", "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "decoding image") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestImage_UnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestImage_ReadBodyError(t *testing.T) {
	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

// imageAPIKeyURLTransport captures the URL for API-key image requests.
type imageAPIKeyURLTransport struct {
	captured string
}

func (tr *imageAPIKeyURLTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.captured = req.URL.String()
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	body := fmt.Sprintf(`{"predictions":[{"bytesBase64Encoded":"%s","mimeType":"image/png"}]}`, imgData)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}, nil
}

func TestImage_APIKeySkipsBearerAuth(t *testing.T) {
	transport := &imageAPIKeyURLTransport{}
	model := Image("imagen-3.0-generate-002",
		WithAPIKey("my-img-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	// URL should contain ?key=
	if !strings.Contains(transport.captured, "?key=my-img-key") {
		t.Errorf("URL should contain ?key=, got %q", transport.captured)
	}
	if !strings.Contains(transport.captured, "generativelanguage.googleapis.com") {
		t.Errorf("URL should use generativelanguage.googleapis.com, got %q", transport.captured)
	}
	if len(result.Images) != 1 {
		t.Errorf("expected 1 image, got %d", len(result.Images))
	}
}

func TestImage_DefaultMimeType(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": ""},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("expected default image/png, got %q", result.Images[0].MediaType)
	}
}

func TestImage_WithHTTPClient(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData, "mimeType": "image/png"},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(srv.URL),
		WithHTTPClient(&http.Client{}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
}
