//go:build e2e

package google_test

import (
	"cmp"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func TestE2E_GoogleVeo_GenerateVideo(t *testing.T) {
	apiKey := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY"))
	if apiKey == "" {
		t.Skip("Google Gemini API credentials not set")
	}
	modelID := os.Getenv("GOOGLE_VEO_MODEL")
	if modelID == "" {
		modelID = "veo-3.1-generate-preview"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()

	result, err := goai.GenerateVideo(ctx, google.Video(modelID, google.WithAPIKey(apiKey)),
		goai.WithVideoPrompt("A single blue paper plane gliding across a plain white background"),
		goai.WithVideoCount(1),
		goai.WithVideoAspectRatio("16:9"),
		goai.WithVideoDuration(4*time.Second),
		goai.WithVideoPollTimeout(10*time.Minute),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Videos) != 1 || len(result.Video.Data) == 0 {
		t.Fatalf("generated videos = %d, first video bytes = %d", len(result.Videos), len(result.Video.Data))
	}
	if !strings.HasPrefix(result.Video.MediaType, "video/") {
		t.Fatalf("media type = %q, want video/*", result.Video.MediaType)
	}
}
