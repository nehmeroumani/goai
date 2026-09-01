package provider_test

import (
	"context"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
	"go.uber.org/goleak"
)

// blockingReadCloser is an io.ReadCloser that can simulate a stalled upstream
// stream. When blocked is true, Read blocks until Close is called (mirroring
// how closing an HTTP response body unblocks a blocked scanner.Read). It
// records how many times Close is invoked so tests can assert single-close.
type blockingReadCloser struct {
	data    []byte
	pos     int
	blocked bool
	unblock chan struct{}
	mu      sync.Mutex
	closed  int
}

func (b *blockingReadCloser) Read(p []byte) (int, error) {
	if b.blocked {
		<-b.unblock // wait for the watchdog to close the body
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pos >= len(b.data) {
		return 0, io.EOF
	}
	n := copy(p, b.data[b.pos:])
	b.pos += n
	return n, nil
}

func (b *blockingReadCloser) Close() error {
	b.mu.Lock()
	b.closed++
	b.mu.Unlock()
	select {
	case <-b.unblock:
	default:
		close(b.unblock)
	}
	return nil
}

func (b *blockingReadCloser) closeCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.closed
}

// TestRunStream_NormalCompletion_ClosesBodyOnce verifies that on a clean
// stream the parse function runs to completion, the output channel is closed,
// and the body is closed exactly once by the deferred closeBody.
func TestRunStream_NormalCompletion_ClosesBodyOnce(t *testing.T) {
	defer goleak.VerifyNone(t)

	body := &blockingReadCloser{data: []byte("hello"), blocked: false, unblock: make(chan struct{})}
	result := provider.RunStream(context.Background(), body, func(_ context.Context, r io.Reader, out chan<- provider.StreamChunk) {
		buf := make([]byte, 5)
		n, _ := r.Read(buf)
		out <- provider.StreamChunk{Type: provider.ChunkText, Text: string(buf[:n])}
		close(out)
	})

	var texts []string
	for ch := range result.Stream {
		if ch.Type == provider.ChunkText {
			texts = append(texts, ch.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "hello" {
		t.Fatalf("got texts %v, want [hello]", texts)
	}

	waitFor(t, func() bool { return body.closeCount() == 1 })
	if got := body.closeCount(); got != 1 {
		t.Fatalf("body closed %d times, want exactly 1", got)
	}
}

// TestRunStream_ContextCancel_UnblocksAndClosesBodyOnce verifies the watchdog:
// cancelling the context must close the body, which unblocks a blocked Read,
// allowing the parse goroutine to exit (no leak), and the body is closed
// exactly once.
func TestRunStream_ContextCancel_UnblocksAndClosesBodyOnce(t *testing.T) {
	defer goleak.VerifyNone(t)

	ctx, cancel := context.WithCancel(context.Background())
	body := &blockingReadCloser{data: []byte("x"), blocked: true, unblock: make(chan struct{})}
	result := provider.RunStream(ctx, body, func(_ context.Context, r io.Reader, out chan<- provider.StreamChunk) {
		buf := make([]byte, 1)
		_, _ = r.Read(buf) // blocks until the watchdog closes the body
		close(out)
	})

	cancel()

	// The watchdog must close the body, unblocking Read so the goroutine exits
	// and the output channel closes.
	select {
	case _, ok := <-result.Stream:
		if ok {
			for range result.Stream { // drain any remaining chunks
			}
		}
	case <-time.After(2 * time.Second):
		t.Fatal("stream not closed after context cancellation; watchdog failed to unblock Read")
	}

	waitFor(t, func() bool { return body.closeCount() == 1 })
	if got := body.closeCount(); got != 1 {
		t.Fatalf("body closed %d times, want exactly 1", got)
	}
}

// TestRunStream_PreCancelledContext verifies the watchdog path when the context
// is already cancelled before RunStream starts: the body is still closed and
// the goroutines exit without leaking.
func TestRunStream_PreCancelledContext(t *testing.T) {
	defer goleak.VerifyNone(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	body := &blockingReadCloser{data: []byte("x"), blocked: true, unblock: make(chan struct{})}
	result := provider.RunStream(ctx, body, func(_ context.Context, r io.Reader, out chan<- provider.StreamChunk) {
		buf := make([]byte, 1)
		_, _ = r.Read(buf)
		close(out)
	})

	for range result.Stream {
	}
	waitFor(t, func() bool { return body.closeCount() == 1 })
	if got := body.closeCount(); got != 1 {
		t.Fatalf("body closed %d times, want exactly 1", got)
	}
}

// TestRunStream_EmptyBody ensures a parse function that reads to EOF and closes
// the channel still results in a single body close (exercising the deferred
// closeBody path with no watchdog involvement).
func TestRunStream_EmptyBody(t *testing.T) {
	defer goleak.VerifyNone(t)

	body := &blockingReadCloser{data: []byte{}, blocked: false, unblock: make(chan struct{})}
	result := provider.RunStream(context.Background(), body, func(_ context.Context, r io.Reader, out chan<- provider.StreamChunk) {
		_, _ = io.ReadAll(strings.NewReader("")) // consume nothing
		_ = r
		close(out)
	})

	for range result.Stream {
	}
	waitFor(t, func() bool { return body.closeCount() == 1 })
	if got := body.closeCount(); got != 1 {
		t.Fatalf("body closed %d times, want exactly 1", got)
	}
}

// waitFor polls cond until it returns true or the deadline passes.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
}