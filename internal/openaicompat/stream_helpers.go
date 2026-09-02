package openaicompat

import (
	"context"
	"io"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// NewSSEStream sets up a standard SSE streaming pipeline for OpenAI-compatible
// providers. It delegates the close-on-cancel watchdog wiring to
// provider.RunStream, which spawns a goroutine that reads SSE events from body
// via ParseStream and handles context cancellation to prevent goroutine leaks.
//
// The returned StreamResult owns the body; callers must not close it.
func NewSSEStream(ctx context.Context, body io.ReadCloser) *provider.StreamResult {
	return provider.RunStream(ctx, body, func(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
		ParseStream(ctx, sse.NewScanner(body), out)
	})
}
