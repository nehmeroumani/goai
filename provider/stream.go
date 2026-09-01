package provider

import (
	"context"
	"io"
	"sync"
)

// RunStream wires a close-on-cancel watchdog around a streaming parse function.
//
// It creates the output channel and spawns a goroutine that runs parse. The
// body is guaranteed to be closed exactly once: either when parse returns (via
// the deferred closeBody) or when ctx is cancelled (via the watchdog goroutine,
// which closes the body to unblock a blocking scanner.Scan() and prevent a
// goroutine leak if the server stalls mid-stream). The closeOnce guards against
// a double close racing between the two paths.
//
// The returned StreamResult owns the body; callers must not close it.
func RunStream(ctx context.Context, body io.ReadCloser, parse func(context.Context, io.Reader, chan<- StreamChunk)) *StreamResult {
	out := make(chan StreamChunk, 64)
	go func() {
		var closeOnce sync.Once
		closeBody := func() { closeOnce.Do(func() { _ = body.Close() }) }
		defer closeBody()
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				closeBody()
			case <-done:
			}
		}()
		parse(ctx, body, out)
	}()
	return &StreamResult{Stream: out}
}
