// Package requesty provides a Requesty language model implementation for GoAI.
//
// Requesty is a unified, OpenAI-compatible LLM gateway that routes to multiple
// providers through a single endpoint, with caching, failover, and cost
// optimization. See https://requesty.ai and https://docs.requesty.ai.
package requesty

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://router.requesty.ai/v1"

// Option configures the Requesty provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

// WithBaseURL overrides the default API base URL.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(h map[string]string) Option {
	return func(o *options) { o.headers = h }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

// Chat creates a Requesty language model for the given model ID.
//
// Model IDs use the provider/model naming convention, e.g.
// "openai/gpt-4o-mini" or "anthropic/claude-sonnet-4-5".
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("REQUESTY_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("REQUESTY_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:        "requesty",
		ModelID:           modelID,
		BaseURL:           o.baseURL,
		TokenSource:       o.tokenSource,
		TokenRequired:     true,
		Headers:           openaicompat.MergeHeaders(o.headers, requestyFixedHeaders),
		HTTPClient:        o.httpClient,
		Capabilities:      chatCaps,
		WarnPromptCaching: true,
		RequestConfig: openaicompat.RequestConfig{
			IncludeStreamOptions: true,
			// Requesty uses flat {"type":"input_file","file_data":...} (item 59).
			FlatInputFile: true,
			ExtraBody:     map[string]any{"usage": map[string]any{"include": true}},
		},
	})
}

// requestyFixedHeaders are the analytics headers Requesty recommends sending on
// every request (https://docs.requesty.ai), merged with user headers via
// openaicompat.MergeHeaders.
var requestyFixedHeaders = map[string]string{
	"HTTP-Referer": "https://github.com/zendev-sh/goai",
	"X-Title":      "goai",
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
