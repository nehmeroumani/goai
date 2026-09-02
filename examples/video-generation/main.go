//go:build ignore

// Example: Google Veo video generation with automatic operation polling.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()

	result, err := goai.GenerateVideo(ctx, google.Video("veo-3.1-generate-preview"),
		goai.WithVideoPrompt("A paper plane taking flight at sunrise, cinematic"),
		goai.WithVideoAspectRatio("16:9"),
		goai.WithVideoResolution("720p"),
		goai.WithVideoDuration(8*time.Second),
		goai.WithVideoAudio(true),
	)
	if err != nil {
		log.Fatal(err)
	}

	if err := os.WriteFile("generated-video.mp4", result.Video.Data, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Generated %d video(s), saved generated-video.mp4\n", len(result.Videos))
}
