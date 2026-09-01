package anthropic

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// largeErrorBody builds a JSON error body whose message field exceeds 1 MiB,
// with a unique tail marker placed far beyond the 1 MiB read cap. If the
// error-body read were unbounded, the marker would survive into the parsed
// error; with the LimitReader cap it must be truncated away.
func largeErrorBody(tailMarker string) []byte {
	var b bytes.Buffer
	b.WriteString(`{"error":{"message":"`)
	b.Write(bytes.Repeat([]byte{'a'}, 1<<20)) // 1 MiB of padding
	b.WriteString(tailMarker)
	b.WriteString(`"}}`)
	return b.Bytes()
}

// Round-4 fix: the upload error-body read must be bounded. A >1 MiB error body
// must be truncated so a marker beyond the 1 MiB cap never surfaces in the
// parsed error, while a normal error still parses.
func TestFileUploader_UploadFile_ErrorBodyBounded(t *testing.T) {
	const tailMarker = "UPLOAD_TAIL_BEYOND_CAP"

	t.Run("large error body is truncated", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write(largeErrorBody(tailMarker))
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
			Reader:    strings.NewReader("abc"),
			Filename:  "test.txt",
			MediaType: "text/plain",
		})
		if err == nil {
			t.Fatal("UploadFile = nil error, want error for non-OK status")
		}
		if strings.Contains(err.Error(), tailMarker) {
			t.Errorf("parsed error leaks tail marker beyond 1 MiB cap: %q", err.Error())
		}
		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error type = %T, want *goai.APIError", err)
		}
		if len(apiErr.ResponseBody) > 1<<20 {
			t.Errorf("ResponseBody length = %d, want <= 1 MiB", len(apiErr.ResponseBody))
		}
	})

	t.Run("normal error still parses", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprint(w, `{"error":{"message":"invalid_request_error: bad upload"}}`)
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
			Reader:    strings.NewReader("abc"),
			Filename:  "test.txt",
			MediaType: "text/plain",
		})
		if err == nil {
			t.Fatal("UploadFile = nil error, want error for non-OK status")
		}
		if !strings.Contains(err.Error(), "invalid_request_error: bad upload") {
			t.Errorf("parsed error = %q, want message from normal error body", err.Error())
		}
	})
}

// Round-4 fix: the delete error-body read must be bounded, mirroring upload.
func TestFileUploader_DeleteFile_ErrorBodyBounded(t *testing.T) {
	const tailMarker = "DELETE_TAIL_BEYOND_CAP"

	t.Run("large error body is truncated", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write(largeErrorBody(tailMarker))
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file_01JAbCdEfGhIjKlMnOpQrStU"})
		if err == nil {
			t.Fatal("DeleteFile = nil error, want error for non-OK status")
		}
		if strings.Contains(err.Error(), tailMarker) {
			t.Errorf("parsed error leaks tail marker beyond 1 MiB cap: %q", err.Error())
		}
		var apiErr *goai.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error type = %T, want *goai.APIError", err)
		}
		if len(apiErr.ResponseBody) > 1<<20 {
			t.Errorf("ResponseBody length = %d, want <= 1 MiB", len(apiErr.ResponseBody))
		}
	})

	t.Run("normal error still parses", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = fmt.Fprint(w, `{"error":{"message":"internal error deleting file"}}`)
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file_01JAbCdEfGhIjKlMnOpQrStU"})
		if err == nil {
			t.Fatal("DeleteFile = nil error, want error for non-OK status")
		}
		if !strings.Contains(err.Error(), "internal error deleting file") {
			t.Errorf("parsed error = %q, want message from normal error body", err.Error())
		}
	})
}

// #11 -- the upload must NOT send a "purpose" form field; only "file" (and
// optionally "expires_in_seconds") are part of the official Files API.
func TestFileUploader_UploadFile_NoPurposeField(t *testing.T) {
	var contentType string
	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contentType = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_nopurpose","type":"file","size_bytes":3,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"test.txt"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("abc"),
		Filename:  "test.txt",
		MediaType: "text/plain",
		Purpose:   "assistants", // must be ignored
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "file_nopurpose" {
		t.Errorf("ref.ID = %q", ref.ID)
	}

	if !strings.HasPrefix(contentType, "multipart/form-data") {
		t.Fatalf("Content-Type = %q, want multipart/form-data", contentType)
	}
	if strings.Contains(body, "purpose") {
		t.Errorf("upload body contains a purpose field: %q", body)
	}
	if !strings.Contains(body, "name=\"file\"") {
		t.Errorf("upload body missing file part: %q", body)
	}
}

// #10 -- created_at is decoded as an RFC3339 string, and size_bytes/mime_type
// are parsed (not bytes/purpose).
func TestFileUploader_UploadFile_ParsesOfficialFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_official","type":"file","size_bytes":42,"mime_type":"application/pdf","created_at":"2026-08-30T12:00:00Z","filename":"doc.pdf"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:   strings.NewReader("fake-pdf"),
		Filename: "doc.pdf",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	// mime_type from the API wins over content sniffing.
	if ref.MediaType != "application/pdf" {
		t.Errorf("ref.MediaType = %q, want application/pdf (from mime_type)", ref.MediaType)
	}
	if ref.ID != "file_official" {
		t.Errorf("ref.ID = %q", ref.ID)
	}
}

// #10 -- a numeric created_at (the old, incorrect wire format) must not break
// decoding; the string field simply won't match a number, so we rely on the
// API returning RFC3339. Here we verify the string field decodes cleanly.
func TestFileUploader_UploadFile_CreatedAtString(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_str","type":"file","size_bytes":3,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"a.txt"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("abc"),
		Filename:  "a.txt",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "file_str" {
		t.Errorf("ref.ID = %q", ref.ID)
	}
}

// When the Files API omits mime_type (empty string), UploadFile must fall back
// to the caller-supplied MediaType before resorting to content sniffing.
func TestFileUploader_UploadFile_MimeFallbackToCallerMediaType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Empty mime_type forces the fallback to upload.MediaType.
		_, _ = fmt.Fprint(w, `{"id":"file_fallback","type":"file","size_bytes":3,"mime_type":"","created_at":"2026-08-30T12:00:00Z","filename":"a.bin"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("binary-data"),
		Filename:  "a.bin",
		MediaType: "application/octet-stream",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.MediaType != "application/octet-stream" {
		t.Errorf("ref.MediaType = %q, want caller-supplied application/octet-stream", ref.MediaType)
	}
}

// A caller-supplied header named "x-api-key" must NOT override the real
// credential on the upload path: auth is applied last, so it wins.
func TestFileUploader_UploadFile_AuthHeaderCannotBeOverridden(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Errorf("x-api-key = %q, want %q (caller header must not win)", got, "test-key")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_auth","type":"file","size_bytes":3,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"test.txt"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL),
		WithHeaders(map[string]string{"x-api-key": "attacker"}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("abc"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
}

// Same invariant on the delete path: auth is applied last, so it wins.
func TestFileUploader_DeleteFile_AuthHeaderCannotBeOverridden(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Errorf("x-api-key = %q, want %q (caller header must not win)", got, "test-key")
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL),
		WithHeaders(map[string]string{"x-api-key": "attacker"}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	if err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file_01JAbCdEfGhIjKlMnOpQrStU"}); err != nil {
		t.Fatalf("DeleteFile error: %v", err)
	}
}

var _ = http.StatusOK

// DeleteFile must reject any ref.ID that could traverse out of the
// /v1/files/ prefix before it is interpolated into the URL path. This covers
// slashes, backslashes, "..", and percent-encoded separators ("%2f").
func TestFileUploader_DeleteFile_RejectsTraversal(t *testing.T) {
	unsafeIDs := []string{
		"../../etc/passwd",
		"..%2f..%2fetc%2fpasswd",
		"file/../admin",
		"file\\..\\admin",
		"file%2f..%2fadmin",
		"%2e%2e",
		"file%2e%2e",
		"..",
		"a%",
	}
	for _, id := range unsafeIDs {
		id := id
		t.Run(id, func(t *testing.T) {
			model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("http://example.invalid"))
			uploader := model.(provider.FileUploadCapableModel).FileUploader()

			err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: id})
			if err == nil {
				t.Fatalf("DeleteFile(%q) = nil error, want rejection", id)
			}
			if !strings.Contains(err.Error(), "unsafe ID") {
				t.Errorf("DeleteFile(%q) error = %v, want 'unsafe ID'", id, err)
			}
		})
	}
}

// A legitimate Anthropic file ID (e.g. "file_01J...") must pass the guard and
// reach the DELETE endpoint, hitting the server with the ID intact in the path.
func TestFileUploader_DeleteFile_AcceptsValidID(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file_01JAbCdEfGhIjKlMnOpQrStU"})
	if err != nil {
		t.Fatalf("DeleteFile valid ID error: %v", err)
	}
	if gotPath != "/v1/files/file_01JAbCdEfGhIjKlMnOpQrStU" {
		t.Errorf("request path = %q, want /v1/files/file_01JAbCdEfGhIjKlMnOpQrStU", gotPath)
	}
}

// Round-5 fix: the upload SUCCESS body read must be bounded. A success body
// larger than 64 MiB must be rejected with an "exceeds" error, while a normal
// success body still parses.
func TestFileUploader_UploadFile_SuccessBodyBounded(t *testing.T) {
	t.Run("oversized success body rejected", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(bytes.Repeat([]byte{' '}, maxFileResponseBytes+2))
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
			Reader:   strings.NewReader("fake-pdf"),
			Filename: "doc.pdf",
		})
		if err == nil {
			t.Fatal("expected error for oversized success body, got nil")
		}
		if !strings.Contains(err.Error(), "exceeds") {
			t.Errorf("err = %v, want substring 'exceeds'", err)
		}
	})

	t.Run("normal success body parses", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"file_ok","type":"file","size_bytes":4,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"a.txt"}`)
		}))
		defer server.Close()

		model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
		uploader := model.(provider.FileUploadCapableModel).FileUploader()

		ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
			Reader:   strings.NewReader("fake"),
			Filename: "a.txt",
		})
		if err != nil {
			t.Fatalf("UploadFile error: %v", err)
		}
		if ref.ID != "file_ok" {
			t.Errorf("ref.ID = %q, want file_ok", ref.ID)
		}
	})
}

// Round-5 fix: an error while reading the upload success body must propagate.
func TestFileUploader_UploadFile_ReadError(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL("http://localhost"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.(provider.FileUploadCapableModel).FileUploader().UploadFile(t.Context(), provider.FileUpload{
		Reader: strings.NewReader("fake"), Filename: "a.txt",
	})
	if err == nil {
		t.Fatal("expected error from read failure")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("expected 'reading response' error, got: %v", err)
	}
}