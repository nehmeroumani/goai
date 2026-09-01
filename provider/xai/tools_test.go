package xai

import (
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// ---------------------------------------------------------------------------
// WebSearch
// ---------------------------------------------------------------------------

func TestTools_WebSearch_Default(t *testing.T) {
	def := Tools.WebSearch()
	if def.Name != "web_search" {
		t.Errorf("Name = %q, want web_search", def.Name)
	}
	if def.ProviderDefinedType != "web_search" {
		t.Errorf("ProviderDefinedType = %q, want web_search", def.ProviderDefinedType)
	}
	if len(def.ProviderDefinedOptions) != 0 {
		t.Errorf("expected empty options, got %v", def.ProviderDefinedOptions)
	}
}

func TestTools_WebSearch_AllOptions(t *testing.T) {
	def := Tools.WebSearch(
		WithAllowedDomains("example.com", "test.org"),
		WithExcludedDomains("spam.com"),
		WithWebSearchImageUnderstanding(true),
	)

	opts := def.ProviderDefinedOptions
	allowed, ok := opts["allowed_domains"].([]string)
	if !ok || len(allowed) != 2 || allowed[0] != "example.com" {
		t.Errorf("allowed_domains = %v", opts["allowed_domains"])
	}
	excluded, ok := opts["excluded_domains"].([]string)
	if !ok || len(excluded) != 1 || excluded[0] != "spam.com" {
		t.Errorf("excluded_domains = %v", opts["excluded_domains"])
	}
	if opts["enable_image_understanding"] != true {
		t.Errorf("enable_image_understanding = %v", opts["enable_image_understanding"])
	}
}

func TestTools_WebSearch_ImageUnderstandingFalse(t *testing.T) {
	// When false, enable_image_understanding should NOT be set.
	def := Tools.WebSearch(WithWebSearchImageUnderstanding(false))
	if _, ok := def.ProviderDefinedOptions["enable_image_understanding"]; ok {
		t.Error("enable_image_understanding should not be set when false")
	}
}

// ---------------------------------------------------------------------------
// XSearch
// ---------------------------------------------------------------------------

func TestTools_XSearch_Default(t *testing.T) {
	def := Tools.XSearch()
	if def.Name != "x_search" {
		t.Errorf("Name = %q, want x_search", def.Name)
	}
	if def.ProviderDefinedType != "x_search" {
		t.Errorf("ProviderDefinedType = %q, want x_search", def.ProviderDefinedType)
	}
	if len(def.ProviderDefinedOptions) != 0 {
		t.Errorf("expected empty options, got %v", def.ProviderDefinedOptions)
	}
}

func TestTools_XSearch_AllOptions(t *testing.T) {
	def := Tools.XSearch(
		WithAllowedXHandles("@alice", "@bob"),
		WithExcludedXHandles("@spam"),
		WithXSearchDateRange("2025-01-01", "2025-12-31"),
		WithXSearchImageUnderstanding(true),
		WithXSearchVideoUnderstanding(true),
	)

	opts := def.ProviderDefinedOptions
	handles, ok := opts["allowed_x_handles"].([]string)
	if !ok || len(handles) != 2 {
		t.Errorf("allowed_x_handles = %v", opts["allowed_x_handles"])
	}
	excluded, ok := opts["excluded_x_handles"].([]string)
	if !ok || len(excluded) != 1 {
		t.Errorf("excluded_x_handles = %v", opts["excluded_x_handles"])
	}
	if opts["from_date"] != "2025-01-01" {
		t.Errorf("from_date = %v", opts["from_date"])
	}
	if opts["to_date"] != "2025-12-31" {
		t.Errorf("to_date = %v", opts["to_date"])
	}
	if opts["enable_image_understanding"] != true {
		t.Errorf("enable_image_understanding = %v", opts["enable_image_understanding"])
	}
	if opts["enable_video_understanding"] != true {
		t.Errorf("enable_video_understanding = %v", opts["enable_video_understanding"])
	}
}

func TestTools_XSearch_PartialDateRange(t *testing.T) {
	// Only from_date set (via empty to_date in range).
	def := Tools.XSearch(WithXSearchDateRange("2025-01-01", ""))
	opts := def.ProviderDefinedOptions
	if opts["from_date"] != "2025-01-01" {
		t.Errorf("from_date = %v", opts["from_date"])
	}
	if _, ok := opts["to_date"]; ok {
		t.Error("to_date should not be set when empty")
	}
}

func TestTools_XSearch_UnderstandingFalse(t *testing.T) {
	def := Tools.XSearch(
		WithXSearchImageUnderstanding(false),
		WithXSearchVideoUnderstanding(false),
	)
	if _, ok := def.ProviderDefinedOptions["enable_image_understanding"]; ok {
		t.Error("enable_image_understanding should not be set when false")
	}
	if _, ok := def.ProviderDefinedOptions["enable_video_understanding"]; ok {
		t.Error("enable_video_understanding should not be set when false")
	}
}

// ---------------------------------------------------------------------------
// Responses-API gate (item 61)
// ---------------------------------------------------------------------------

// plainChatRequest exercises DoGenerate with a plain text request and no tools.
func plainChatRequest(t *testing.T) error {
	t.Helper()
	model := Chat("grok-3", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	// We expect a connection error (nothing listens on :1), which proves the
	// request reached the HTTP layer rather than being gated.
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "Responses API") {
		t.Errorf("plain chat should not be gated, got: %v", err)
	}
	return err
}

func TestChat_PlainTextNotGated(t *testing.T) {
	err := plainChatRequest(t)
	if err == nil || !strings.Contains(err.Error(), "sending request") {
		t.Errorf("plain chat should reach HTTP layer, got: %v", err)
	}
}

func TestChat_GatesWebSearch(t *testing.T) {
	model := Chat("grok-3", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "latest news"}}},
		},
		Tools: []provider.ToolDefinition{Tools.WebSearch()},
	})
	if err == nil {
		t.Fatal("expected error for web_search on Chat Completions path")
	}
	if !strings.Contains(err.Error(), "web_search") || !strings.Contains(err.Error(), "Responses API") {
		t.Errorf("error should mention web_search and Responses API, got: %v", err)
	}
}

func TestChat_GatesXSearch(t *testing.T) {
	model := Chat("grok-3", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "x posts"}}},
		},
		Tools: []provider.ToolDefinition{Tools.XSearch()},
	})
	if err == nil {
		t.Fatal("expected error for x_search on Chat Completions path")
	}
	if !strings.Contains(err.Error(), "x_search") || !strings.Contains(err.Error(), "Responses API") {
		t.Errorf("error should mention x_search and Responses API, got: %v", err)
	}
}

func TestChat_GatesWebSearchOnStream(t *testing.T) {
	model := Chat("grok-3", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "latest news"}}},
		},
		Tools: []provider.ToolDefinition{Tools.WebSearch()},
	})
	if err == nil {
		t.Fatal("expected error for web_search on streaming Chat Completions path")
	}
	if !strings.Contains(err.Error(), "Responses API") {
		t.Errorf("error should mention Responses API, got: %v", err)
	}
}

func TestChat_RegularFunctionToolNotGated(t *testing.T) {
	// A regular (non-provider-defined) function tool must NOT be gated.
	err := plainChatRequest(t)
	if err == nil {
		t.Fatal("expected error")
	}
	model := Chat("grok-3", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: []byte(`{"type":"object"}`)},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "Responses API") {
		t.Errorf("regular function tool should not be gated, got: %v", err)
	}
}
