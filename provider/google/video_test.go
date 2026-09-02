package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

func TestVideo_GenerateIntegration(t *testing.T) {
	var polls atomic.Int32
	var serverURL string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("x-goog-api-key = %q", r.Header.Get("x-goog-api-key"))
		}

		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1beta/models/veo-3.1-generate-preview:predictLongRunning":
			if r.Method != http.MethodPost {
				t.Errorf("method = %s, want POST", r.Method)
			}
			var body struct {
				Instances []struct {
					Prompt string `json:"prompt"`
					Image  *struct {
						BytesBase64Encoded string `json:"bytesBase64Encoded"`
						MimeType           string `json:"mimeType"`
					} `json:"image"`
				} `json:"instances"`
				Parameters map[string]any `json:"parameters"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if len(body.Instances) != 1 || body.Instances[0].Prompt != "a paper plane taking flight" {
				t.Errorf("instances = %+v", body.Instances)
			}
			if got := body.Instances[0].Image; got == nil || got.BytesBase64Encoded != base64.StdEncoding.EncodeToString([]byte("png")) || got.MimeType != "image/png" {
				t.Errorf("image = %+v", got)
			}
			if body.Parameters["sampleCount"] != float64(1) || body.Parameters["aspectRatio"] != "16:9" {
				t.Errorf("parameters = %+v", body.Parameters)
			}
			if body.Parameters["durationSeconds"] != float64(8) || body.Parameters["generateAudio"] != true {
				t.Errorf("parameters = %+v", body.Parameters)
			}
			_, _ = fmt.Fprint(w, `{"name":"models/veo-3.1-generate-preview/operations/op-123"}`)

		case "/v1beta/models/veo-3.1-generate-preview/operations/op-123":
			if r.Method != http.MethodGet {
				t.Errorf("method = %s, want GET", r.Method)
			}
			if polls.Add(1) == 1 {
				_, _ = fmt.Fprint(w, `{"name":"models/veo-3.1-generate-preview/operations/op-123","done":false}`)
				return
			}
			_, _ = fmt.Fprintf(w, `{"name":"models/veo-3.1-generate-preview/operations/op-123","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/download/video.mp4")

		case "/download/video.mp4":
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("fake-mp4"))

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	result, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("a paper plane taking flight"),
		goai.WithVideoImage(provider.MediaData{Data: []byte("png"), MediaType: "image/png"}),
		goai.WithVideoAspectRatio("16:9"),
		goai.WithVideoDuration(8*time.Second),
		goai.WithVideoAudio(true),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	if polls.Load() != 2 {
		t.Errorf("polls = %d, want 2", polls.Load())
	}
	if len(result.Videos) != 1 || string(result.Video.Data) != "fake-mp4" {
		t.Fatalf("videos = %+v", result.Videos)
	}
	if result.Video.MediaType != "video/mp4" {
		t.Errorf("media type = %q", result.Video.MediaType)
	}
	if result.Response.ID != "models/veo-3.1-generate-preview/operations/op-123" {
		t.Errorf("response ID = %q", result.Response.ID)
	}
}

func TestVideo_OperationError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost {
			_, _ = fmt.Fprint(w, `{"name":"operations/failed"}`)
			return
		}
		_, _ = fmt.Fprint(w, `{"name":"operations/failed","done":true,"error":{"code":13,"message":"generation failed"}}`)
	}))
	defer server.Close()

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
		goai.WithVideoMaxRetries(0),
	)
	if err == nil || err.Error() != "google video generation failed: generation failed (code 13)" {
		t.Fatalf("error = %v", err)
	}
}

func TestVideo_DownloadRedirectDoesNotLeakAPIKey(t *testing.T) {
	var leakedKey string
	downloadServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leakedKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("video"))
	}))
	defer downloadServer.Close()

	var apiServerURL string
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("API request key = %q", r.Header.Get("x-goog-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			_, _ = fmt.Fprint(w, `{"name":"operations/redirect"}`)
		case http.MethodGet:
			if r.URL.Path == "/download/redirect" {
				http.Redirect(w, r, downloadServer.URL+"/video.mp4", http.StatusFound)
				return
			}
			_, _ = fmt.Fprintf(w, `{"name":"operations/redirect","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, apiServerURL+"/download/redirect")
		}
	}))
	defer apiServer.Close()
	apiServerURL = apiServer.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(apiServer.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err == nil || !strings.Contains(err.Error(), "cross-origin redirect") {
		t.Fatalf("error = %v, want cross-origin redirect error", err)
	}
	if leakedKey != "" {
		t.Fatalf("API key leaked to redirected origin: %q", leakedKey)
	}
}

func TestVideo_AllowsSameOriginDownloadRedirect(t *testing.T) {
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("x-goog-api-key = %q", r.Header.Get("x-goog-api-key"))
		}
		switch {
		case r.Method == http.MethodPost:
			_, _ = fmt.Fprint(w, `{"name":"operations/same-origin-download"}`)
		case r.URL.Path == "/v1beta/operations/same-origin-download":
			_, _ = fmt.Fprintf(w, `{"name":"operations/same-origin-download","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/download/start")
		case r.URL.Path == "/download/start":
			http.Redirect(w, r, serverURL+"/download/final", http.StatusFound)
		case r.URL.Path == "/download/final":
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("video"))
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(result.Video.Data) != "video" {
		t.Fatalf("video = %q", result.Video.Data)
	}
}

func TestVideo_RejectsOperationRedirect(t *testing.T) {
	var redirectedCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirected-start" {
			redirectedCalls.Add(1)
			_, _ = fmt.Fprint(w, `{"name":"operations/unsafe"}`)
			return
		}
		http.Redirect(w, r, "/redirected-start", http.StatusTemporaryRedirect)
	}))
	defer server.Close()

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model, goai.WithVideoPrompt("test"))
	if err == nil {
		t.Fatal("expected redirect error")
	}
	if redirectedCalls.Load() != 0 {
		t.Fatalf("redirected operation endpoint called %d times", redirectedCalls.Load())
	}
}

func TestVideo_TransientPollFailureDoesNotRestartGeneration(t *testing.T) {
	var starts atomic.Int32
	var polls atomic.Int32
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			starts.Add(1)
			_, _ = fmt.Fprint(w, `{"name":"operations/retry-poll"}`)
		case r.URL.Path == "/v1beta/operations/retry-poll":
			if polls.Add(1) == 1 {
				w.Header().Set("retry-after-ms", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = fmt.Fprint(w, `{"error":{"message":"try again"}}`)
				return
			}
			_, _ = fmt.Fprintf(w, `{"name":"operations/retry-poll","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/video.mp4")
		case r.URL.Path == "/video.mp4":
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("video"))
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	if starts.Load() != 1 {
		t.Fatalf("generation starts = %d, want 1", starts.Load())
	}
	if polls.Load() != 2 {
		t.Fatalf("polls = %d, want 2", polls.Load())
	}
}

func TestVideo_RetriesExplicitStartFailure(t *testing.T) {
	var starts atomic.Int32
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			if starts.Add(1) == 1 {
				w.Header().Set("retry-after-ms", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = fmt.Fprint(w, `{"error":{"message":"try again"}}`)
				return
			}
			_, _ = fmt.Fprint(w, `{"name":"operations/retry-start"}`)
		case r.URL.Path == "/v1beta/operations/retry-start":
			_, _ = fmt.Fprintf(w, `{"name":"operations/retry-start","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/video.mp4")
		case r.URL.Path == "/video.mp4":
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("video"))
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoMaxRetries(1),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
	if starts.Load() != 2 {
		t.Fatalf("starts = %d, want 2", starts.Load())
	}
}

func TestVideo_DoesNotRetryAmbiguousStartFailure(t *testing.T) {
	var starts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		starts.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":{"message":"response lost after enqueue may be ambiguous"}}`)
	}))
	defer server.Close()

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoMaxRetries(2),
	)
	if err == nil {
		t.Fatal("expected start error")
	}
	if starts.Load() != 1 {
		t.Fatalf("generation starts = %d, want 1", starts.Load())
	}
}

func TestVideo_TransientDownloadFailureDoesNotRestartGeneration(t *testing.T) {
	var starts atomic.Int32
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			starts.Add(1)
			_, _ = fmt.Fprint(w, `{"name":"operations/download-failure"}`)
		case r.URL.Path == "/v1beta/operations/download-failure":
			_, _ = fmt.Fprintf(w, `{"name":"operations/download-failure","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/video.mp4")
		case r.URL.Path == "/video.mp4":
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = fmt.Fprint(w, `{"error":{"message":"temporarily unavailable"}}`)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoMaxRetries(2),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err == nil {
		t.Fatal("expected download error")
	}
	if starts.Load() != 1 {
		t.Fatalf("generation starts = %d, want 1", starts.Load())
	}
}

func TestVideo_RejectsCrossOriginOperationURL(t *testing.T) {
	var externalCalls atomic.Int32
	external := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		externalCalls.Add(1)
		_, _ = fmt.Fprint(w, `{"done":false}`)
	}))
	defer external.Close()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintf(w, `{"name":%q}`, external.URL+"/operations/unsafe")
	}))
	defer server.Close()

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err == nil || !strings.Contains(err.Error(), "configured API origin") {
		t.Fatalf("error = %v, want origin error", err)
	}
	if externalCalls.Load() != 0 {
		t.Fatalf("external operation endpoint called %d times", externalCalls.Load())
	}
}

func TestVideo_RejectsOversizedDownload(t *testing.T) {
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			_, _ = fmt.Fprint(w, `{"name":"operations/oversized"}`)
		case r.URL.Path == "/v1beta/operations/oversized":
			_, _ = fmt.Fprintf(w, `{"name":"operations/oversized","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q}}]}}}`, serverURL+"/video.mp4")
		case r.URL.Path == "/video.mp4":
			w.Header().Set("Content-Type", "video/mp4")
			w.Header().Set("Content-Length", fmt.Sprint(maxGoogleVideoDownloadBytes+1))
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(time.Second),
	)
	if err == nil || !strings.Contains(err.Error(), "download limit") {
		t.Fatalf("error = %v, want download limit error", err)
	}
}

func TestVideo_LimitsDownloadErrorBody(t *testing.T) {
	body := &countingVideoBody{remaining: 2 << 20}
	model := &videoModel{opts: options{
		baseURL:     "https://video.example.test",
		tokenSource: provider.StaticToken("key"),
		httpClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header:     make(http.Header),
				Body:       body,
			}, nil
		})},
	}}

	_, err := model.downloadVideos(t.Context(), "key", []googleVideoFile{{URI: "https://video.example.test/video.mp4"}})
	if err == nil {
		t.Fatal("expected download error")
	}
	if body.read > (1<<20)+1 {
		t.Fatalf("read %d error-body bytes, want at most %d", body.read, (1<<20)+1)
	}
}

func TestVideo_DownloadResponseEdges(t *testing.T) {
	tests := []struct {
		name      string
		response  *http.Response
		wantError string
		wantType  string
	}{
		{
			name: "error response read failure",
			response: &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header:     make(http.Header),
				Body:       &failReader{},
			},
			wantError: "error response",
		},
		{
			name: "successful response read failure",
			response: &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       &failReader{},
			},
			wantError: "reading video",
		},
		{
			name: "media type fallback",
			response: &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader("video")),
			},
			wantType: "video/webm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &videoModel{opts: options{
				baseURL: "https://video.example.test",
				httpClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					return tt.response, nil
				})},
			}}
			videos, err := model.downloadVideos(t.Context(), "key", []googleVideoFile{{
				URI:      "https://video.example.test/video",
				MimeType: "video/webm",
			}})
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("error = %v, want %q", err, tt.wantError)
				}
				return
			}
			if err != nil || len(videos) != 1 || videos[0].MediaType != tt.wantType {
				t.Fatalf("videos = %#v, error = %v", videos, err)
			}
		})
	}
}

func TestVideo_CancelsStartRequest(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	model := Video(
		"veo-test",
		WithAPIKey("key"),
		WithBaseURL("https://video.example.test"),
		WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			<-req.Context().Done()
			return nil, req.Context().Err()
		})}),
	)
	_, err := model.DoGenerate(ctx, provider.VideoParams{Prompt: "test", N: 1})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want deadline exceeded", err)
	}
}

func TestVideo_CancelsDownloadRequest(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	model := Video(
		"veo-test",
		WithAPIKey("key"),
		WithBaseURL("https://video.example.test"),
		WithHTTPClient(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			<-req.Context().Done()
			return nil, req.Context().Err()
		})}),
	).(*videoModel)
	_, err := model.downloadVideos(ctx, "key", []googleVideoFile{{URI: "https://video.example.test/video.mp4"}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want deadline exceeded", err)
	}
}

type countingVideoBody struct {
	remaining int64
	read      int64
}

func (b *countingVideoBody) Read(p []byte) (int, error) {
	if b.remaining == 0 {
		return 0, io.EOF
	}
	n := int64(len(p))
	if n > b.remaining {
		n = b.remaining
	}
	for i := int64(0); i < n; i++ {
		p[i] = 'x'
	}
	b.remaining -= n
	b.read += n
	return int(n), nil
}

func (*countingVideoBody) Close() error { return nil }

func TestVideo_PollTimeoutBoundsInFlightRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = fmt.Fprint(w, `{"name":"operations/stalled"}`)
			return
		}
		<-r.Context().Done()
	}))
	defer server.Close()

	model := Video("veo-3.1-generate-preview", WithAPIKey("test-key"), WithBaseURL(server.URL))
	started := time.Now()
	_, err := goai.GenerateVideo(t.Context(), model,
		goai.WithVideoPrompt("test"),
		goai.WithVideoPollInterval(time.Millisecond),
		goai.WithVideoPollTimeout(20*time.Millisecond),
	)
	if err == nil || !strings.Contains(err.Error(), "timed out after 20ms") {
		t.Fatalf("error = %v, want poll timeout", err)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("poll timeout took %v", elapsed)
	}
}

func TestVideoRetryDelay_UsesExponentialBackoff(t *testing.T) {
	apiErr := &goai.APIError{StatusCode: http.StatusServiceUnavailable, IsRetryable: true}
	if first, second := videoRetryDelay(apiErr, 0), videoRetryDelay(apiErr, 1); first != 2*time.Second || second != 4*time.Second {
		t.Fatalf("retry delays = %v, %v; want 2s, 4s", first, second)
	}
}

func TestVideoPollRetryDelay_EscalatesPastSmallServerHint(t *testing.T) {
	apiErr := &goai.APIError{
		StatusCode:      http.StatusTooManyRequests,
		IsRetryable:     true,
		ResponseHeaders: map[string]string{"retry-after-ms": "1"},
	}
	if first, second := videoPollRetryDelay(apiErr, 0), videoPollRetryDelay(apiErr, 1); first != 2*time.Second || second != 4*time.Second {
		t.Fatalf("poll retry delays = %v, %v; want 2s, 4s", first, second)
	}
}
