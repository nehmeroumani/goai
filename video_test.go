package goai

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

type mockVideoModel struct {
	generateFn func(ctx context.Context, params provider.VideoParams) (*provider.VideoResult, error)
}

func (m *mockVideoModel) ModelID() string { return "test-video-model" }

func (m *mockVideoModel) DoGenerate(ctx context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
	return m.generateFn(ctx, params)
}

func TestGenerateVideo_NilModel(t *testing.T) {
	_, err := GenerateVideo(t.Context(), nil, WithVideoPrompt("test"))
	if err == nil || err.Error() != "goai: model must not be nil" {
		t.Fatalf("error = %v, want nil model error", err)
	}
}

func TestGenerateVideo_Defaults(t *testing.T) {
	model := &mockVideoModel{
		generateFn: func(_ context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
			if params.Prompt != "a cat walking" {
				t.Errorf("prompt = %q", params.Prompt)
			}
			if params.N != 1 {
				t.Errorf("n = %d, want 1", params.N)
			}
			if params.PollInterval != 5*time.Second {
				t.Errorf("poll interval = %v, want 5s", params.PollInterval)
			}
			if params.PollTimeout != 10*time.Minute {
				t.Errorf("poll timeout = %v, want 10m", params.PollTimeout)
			}
			return &provider.VideoResult{
				Videos: []provider.VideoData{{Data: []byte("fake-mp4"), MediaType: "video/mp4"}},
			}, nil
		},
	}

	result, err := GenerateVideo(t.Context(), model, WithVideoPrompt("a cat walking"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Videos) != 1 || string(result.Videos[0].Data) != "fake-mp4" {
		t.Fatalf("videos = %+v", result.Videos)
	}
	if result.Video.MediaType != "video/mp4" {
		t.Errorf("video media type = %q", result.Video.MediaType)
	}
}

func TestGenerateVideo_Options(t *testing.T) {
	seed := int64(42)
	inputImage := provider.MediaData{Data: []byte("image"), MediaType: "image/png"}
	firstFrame := provider.VideoFrame{Image: inputImage, Type: provider.VideoFrameFirst}
	reference := provider.MediaData{URL: "https://example.com/reference.mp4", MediaType: "video/mp4"}

	model := &mockVideoModel{
		generateFn: func(_ context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
			if params.N != 3 || params.AspectRatio != "16:9" || params.Resolution != "1280x720" {
				t.Errorf("unexpected dimensions/count: %+v", params)
			}
			if params.Duration != 8*time.Second || params.FPS != 24 {
				t.Errorf("duration/fps = %v/%d", params.Duration, params.FPS)
			}
			if params.Seed == nil || *params.Seed != seed {
				t.Errorf("seed = %v", params.Seed)
			}
			if params.GenerateAudio == nil || !*params.GenerateAudio {
				t.Errorf("generate audio = %v", params.GenerateAudio)
			}
			if params.Image == nil || string(params.Image.Data) != "image" {
				t.Errorf("image = %+v", params.Image)
			}
			if len(params.FrameImages) != 1 || params.FrameImages[0].Type != provider.VideoFrameFirst {
				t.Errorf("frame images = %+v", params.FrameImages)
			}
			if len(params.InputReferences) != 1 || params.InputReferences[0].URL != reference.URL {
				t.Errorf("references = %+v", params.InputReferences)
			}
			if params.ProviderOptions["google"].(map[string]any)["negativePrompt"] != "rain" {
				t.Errorf("provider options = %+v", params.ProviderOptions)
			}
			if params.PollInterval != time.Millisecond || params.PollTimeout != time.Second {
				t.Errorf("poll interval/timeout = %v/%v", params.PollInterval, params.PollTimeout)
			}
			return &provider.VideoResult{
				Videos: []provider.VideoData{{Data: []byte("video"), MediaType: "video/mp4"}},
			}, nil
		},
	}

	_, err := GenerateVideo(t.Context(), model,
		WithVideoPrompt("test"),
		WithVideoCount(3),
		WithVideoAspectRatio("16:9"),
		WithVideoResolution("1280x720"),
		WithVideoDuration(8*time.Second),
		WithVideoFPS(24),
		WithVideoSeed(seed),
		WithVideoAudio(true),
		WithVideoImage(inputImage),
		WithVideoFrameImages(firstFrame),
		WithVideoInputReferences(reference),
		WithVideoProviderOptions(map[string]any{"google": map[string]any{"negativePrompt": "rain"}}),
		WithVideoPollInterval(time.Millisecond),
		WithVideoPollTimeout(time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateVideo_Timeout(t *testing.T) {
	model := &mockVideoModel{
		generateFn: func(ctx context.Context, _ provider.VideoParams) (*provider.VideoResult, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}

	_, err := GenerateVideo(t.Context(), model,
		WithVideoPrompt("test"),
		WithVideoTimeout(20*time.Millisecond),
	)
	if err == nil || err != context.DeadlineExceeded {
		t.Fatalf("error = %v, want deadline exceeded", err)
	}
}

func TestGenerateVideo_PassesRetryBudgetToProvider(t *testing.T) {
	model := &mockVideoModel{
		generateFn: func(_ context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
			if params.MaxRetries != 1 {
				t.Errorf("max retries = %d, want 1", params.MaxRetries)
			}
			return &provider.VideoResult{Videos: []provider.VideoData{{Data: []byte("video")}}}, nil
		},
	}

	_, err := GenerateVideo(t.Context(), model,
		WithVideoPrompt("test"),
		WithVideoMaxRetries(1),
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateVideo_ErrorAndEmptyResult(t *testing.T) {
	t.Run("provider error", func(t *testing.T) {
		want := errors.New("provider failed")
		model := &mockVideoModel{generateFn: func(context.Context, provider.VideoParams) (*provider.VideoResult, error) {
			return nil, want
		}}
		if _, err := GenerateVideo(t.Context(), model); !errors.Is(err, want) {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("empty result", func(t *testing.T) {
		model := &mockVideoModel{generateFn: func(context.Context, provider.VideoParams) (*provider.VideoResult, error) {
			return &provider.VideoResult{}, nil
		}}
		if _, err := GenerateVideo(t.Context(), model); err == nil || err.Error() != "goai: no video generated" {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("nil result", func(t *testing.T) {
		model := &mockVideoModel{generateFn: func(context.Context, provider.VideoParams) (*provider.VideoResult, error) {
			return nil, nil
		}}
		if _, err := GenerateVideo(t.Context(), model); err == nil || err.Error() != "goai: no video generated" {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestVideoOptionClamps(t *testing.T) {
	model := &mockVideoModel{generateFn: func(_ context.Context, params provider.VideoParams) (*provider.VideoResult, error) {
		if params.N != 1 || params.MaxRetries != 0 {
			t.Fatalf("params = %#v", params)
		}
		if params.PollInterval != defaultVideoPollInterval || params.PollTimeout != defaultVideoPollTimeout {
			t.Fatalf("poll defaults = %v, %v", params.PollInterval, params.PollTimeout)
		}
		return &provider.VideoResult{Videos: []provider.VideoData{{Data: []byte("video")}}}, nil
	}}
	_, err := GenerateVideo(t.Context(), model,
		WithVideoCount(0),
		WithVideoMaxRetries(-1),
		WithVideoPollInterval(0),
		WithVideoPollTimeout(0),
	)
	if err != nil {
		t.Fatal(err)
	}
}
