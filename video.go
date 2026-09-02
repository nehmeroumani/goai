package goai

import (
	"context"
	"errors"
	"time"

	"github.com/zendev-sh/goai/provider"
)

const (
	defaultVideoPollInterval = 5 * time.Second
	defaultVideoPollTimeout  = 10 * time.Minute
)

// GenerateVideo generates videos from text and media prompts.
func GenerateVideo(ctx context.Context, model provider.VideoModel, opts ...VideoOption) (*VideoResult, error) {
	if model == nil {
		return nil, errors.New("goai: model must not be nil")
	}

	o := videoOptions{
		n:            1,
		maxRetries:   2,
		pollInterval: defaultVideoPollInterval,
		pollTimeout:  defaultVideoPollTimeout,
	}
	for _, opt := range opts {
		opt(&o)
	}

	if o.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.timeout)
		defer cancel()
	}

	params := provider.VideoParams{
		Prompt:          o.prompt,
		Image:           o.image,
		N:               o.n,
		AspectRatio:     o.aspectRatio,
		Resolution:      o.resolution,
		Duration:        o.duration,
		FPS:             o.fps,
		Seed:            o.seed,
		FrameImages:     o.frameImages,
		InputReferences: o.inputReferences,
		GenerateAudio:   o.generateAudio,
		ProviderOptions: o.providerOptions,
		MaxRetries:      o.maxRetries,
		PollInterval:    o.pollInterval,
		PollTimeout:     o.pollTimeout,
	}

	result, err := model.DoGenerate(ctx, params)
	if err != nil {
		return nil, err
	}
	if result == nil || len(result.Videos) == 0 {
		return nil, errors.New("goai: no video generated")
	}

	return &VideoResult{
		Video:            result.Videos[0],
		Videos:           result.Videos,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}, nil
}

// VideoResult contains generated videos and provider response metadata.
type VideoResult struct {
	Video            provider.VideoData
	Videos           []provider.VideoData
	ProviderMetadata map[string]map[string]any
	Response         provider.ResponseMetadata
}

// VideoOption configures a GenerateVideo call.
type VideoOption func(*videoOptions)

type videoOptions struct {
	prompt          string
	image           *provider.MediaData
	n               int
	aspectRatio     string
	resolution      string
	duration        time.Duration
	fps             int
	seed            *int64
	frameImages     []provider.VideoFrame
	inputReferences []provider.MediaData
	generateAudio   *bool
	providerOptions map[string]any
	maxRetries      int
	timeout         time.Duration
	pollInterval    time.Duration
	pollTimeout     time.Duration
}

func WithVideoPrompt(prompt string) VideoOption {
	return func(o *videoOptions) { o.prompt = prompt }
}

// WithVideoImage sets the initial image for image-to-video generation.
func WithVideoImage(image provider.MediaData) VideoOption {
	return func(o *videoOptions) { o.image = &image }
}

// WithVideoCount sets the number of videos to generate.
func WithVideoCount(n int) VideoOption {
	return func(o *videoOptions) {
		if n < 1 {
			n = 1
		}
		o.n = n
	}
}

// WithVideoAspectRatio sets the output aspect ratio, such as "16:9".
func WithVideoAspectRatio(ratio string) VideoOption {
	return func(o *videoOptions) { o.aspectRatio = ratio }
}

// WithVideoResolution sets the provider-supported output resolution.
func WithVideoResolution(resolution string) VideoOption {
	return func(o *videoOptions) { o.resolution = resolution }
}

// WithVideoDuration sets the requested video duration.
func WithVideoDuration(duration time.Duration) VideoOption {
	return func(o *videoOptions) { o.duration = duration }
}

// WithVideoFPS sets the requested frames per second when supported.
func WithVideoFPS(fps int) VideoOption {
	return func(o *videoOptions) { o.fps = fps }
}

// WithVideoSeed sets a deterministic seed when supported.
func WithVideoSeed(seed int64) VideoOption {
	return func(o *videoOptions) { o.seed = &seed }
}

// WithVideoFrameImages sets role-tagged first and last frame images.
func WithVideoFrameImages(frames ...provider.VideoFrame) VideoOption {
	return func(o *videoOptions) {
		o.frameImages = append([]provider.VideoFrame(nil), frames...)
	}
}

// WithVideoInputReferences sets reference images or videos for supporting models.
func WithVideoInputReferences(references ...provider.MediaData) VideoOption {
	return func(o *videoOptions) {
		o.inputReferences = append([]provider.MediaData(nil), references...)
	}
}

// WithVideoAudio controls whether the model generates audio with the video.
func WithVideoAudio(enabled bool) VideoOption {
	return func(o *videoOptions) { o.generateAudio = &enabled }
}

// WithVideoProviderOptions sets provider-specific video parameters.
func WithVideoProviderOptions(opts map[string]any) VideoOption {
	validateProviderOptions("WithVideoProviderOptions", opts)
	return func(o *videoOptions) { o.providerOptions = opts }
}

// WithVideoMaxRetries sets retries for rate-limited generation-start requests.
func WithVideoMaxRetries(n int) VideoOption {
	return func(o *videoOptions) {
		if n < 0 {
			n = 0
		}
		o.maxRetries = n
	}
}

// WithVideoTimeout sets an overall timeout for video generation.
func WithVideoTimeout(timeout time.Duration) VideoOption {
	return func(o *videoOptions) { o.timeout = timeout }
}

// WithVideoPollInterval sets the interval between operation status checks.
func WithVideoPollInterval(interval time.Duration) VideoOption {
	return func(o *videoOptions) {
		if interval > 0 {
			o.pollInterval = interval
		}
	}
}

// WithVideoPollTimeout sets the maximum time spent polling an operation.
func WithVideoPollTimeout(timeout time.Duration) VideoOption {
	return func(o *videoOptions) {
		if timeout > 0 {
			o.pollTimeout = timeout
		}
	}
}
