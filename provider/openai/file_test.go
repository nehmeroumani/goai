package openai

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// TestFile_UploadErrorBodyBounded verifies the error-path bounded read in
// UploadFile: an error response body larger than maxFileErrorBytes is
// truncated, so the tail marker never reaches the extracted error message.
func TestFile_UploadErrorBodyBounded(t *testing.T) {
	const tailMarker = "TAIL-MARKER-SHOULD-NOT-APPEAR"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.Copy(w, io.MultiReader(
			strings.NewReader(`{"error":{"message":"`),
			io.LimitReader(zeroReader{}, int64(maxFileErrorBytes+len(tailMarker))),
			strings.NewReader(tailMarker),
		))
	}))
	defer server.Close()

	u := &fileUploader{opts: options{baseURL: server.URL, tokenSource: provider.StaticToken("k")}}
	_, err := u.UploadFile(t.Context(), provider.FileUpload{
		Reader:   strings.NewReader("file contents"),
		Filename: "notes.txt",
		Purpose:  "assistants",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *goai.APIError", err)
	}
	if strings.Contains(apiErr.Message, tailMarker) {
		t.Errorf("error message contains tail marker; error body was not bounded: %q", apiErr.Message)
	}
}

// TestFile_UploadSuccessBodyBounded verifies the success-path bounded read in
// UploadFile: a success body larger than maxFileResponseBytes is rejected with
// an error mentioning the byte limit, instead of being read unbounded.
func TestFile_UploadSuccessBodyBounded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Write a valid JSON prefix followed by padding that pushes the body
		// past maxFileResponseBytes. The JSON prefix alone is well-formed; the
		// overflow padding guarantees the read exceeds the bound.
		prefix := []byte(`{"id":"file-abc","bytes":12,"created_at":0,"filename":"notes.txt","purpose":"assistants"}`)
		_, _ = w.Write(prefix)
		padding := int(maxFileResponseBytes) + 1 - len(prefix)
		if padding > 0 {
			_, _ = io.Copy(w, io.LimitReader(zeroReader{}, int64(padding)))
		}
	}))
	defer server.Close()

	u := &fileUploader{opts: options{baseURL: server.URL, tokenSource: provider.StaticToken("k")}}
	_, err := u.UploadFile(t.Context(), provider.FileUpload{
		Reader:   strings.NewReader("file contents"),
		Filename: "notes.txt",
		Purpose:  "assistants",
	})
	if err == nil {
		t.Fatal("expected error for oversized success body")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error = %q, want it to mention the size limit", err)
	}
}

// TestFile_UploadSuccessBodyParses verifies that a normal-sized success body
// still decodes into a usable RemoteFileRef.
func TestFile_UploadSuccessBodyParses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"file-abc","bytes":12,"created_at":1700000000,"filename":"notes.txt","purpose":"assistants"}`))
	}))
	defer server.Close()

	u := &fileUploader{opts: options{baseURL: server.URL, tokenSource: provider.StaticToken("k")}}
	ref, err := u.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("file contents"),
		Filename:  "notes.txt",
		Purpose:   "assistants",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ref.ID != "file-abc" {
		t.Errorf("ref.ID = %q, want %q", ref.ID, "file-abc")
	}
	if ref.Filename != "notes.txt" {
		t.Errorf("ref.Filename = %q, want %q", ref.Filename, "notes.txt")
	}
}

// TestFile_DeleteErrorBodyBounded verifies the error-path bounded read in
// DeleteFile.
func TestFile_DeleteErrorBodyBounded(t *testing.T) {
	const tailMarker = "TAIL-MARKER-SHOULD-NOT-APPEAR"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.Copy(w, io.MultiReader(
			strings.NewReader(`{"error":{"message":"`),
			io.LimitReader(zeroReader{}, int64(maxFileErrorBytes+len(tailMarker))),
			strings.NewReader(tailMarker),
		))
	}))
	defer server.Close()

	u := &fileUploader{opts: options{baseURL: server.URL, tokenSource: provider.StaticToken("k")}}
	err := u.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-abc123"})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *goai.APIError", err)
	}
	if strings.Contains(apiErr.Message, tailMarker) {
		t.Errorf("error message contains tail marker; error body was not bounded: %q", apiErr.Message)
	}
}
