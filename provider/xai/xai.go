// Package xai provides a xai language model implementation for GoAI.
package xai

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://api.x.ai/v1"

// Option configures the xai provider.
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

// Chat creates a xai language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("XAI_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("XAI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	inner := openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:        "xai",
		ModelID:           modelID,
		BaseURL:           o.baseURL,
		TokenSource:       o.tokenSource,
		TokenRequired:     true,
		Headers:           o.headers,
		HTTPClient:        o.httpClient,
		Capabilities:      chatCaps,
		WarnPromptCaching: true,
		RequestConfig: openaicompat.RequestConfig{
			IncludeStreamOptions:    true,
			IncludeReasoningContent: true, // xAI reasoning round-trip (item 60).
		},
	})
	// xAI exposes its server tools (web_search, x_search) only through the
	// Responses API. This provider is a Chat Completions wrapper, so those
	// tools are gated here with a clear error (item 61).
	return &responsesAPIGatedModel{LanguageModel: inner}
}

// responsesAPIGatedModel wraps a Chat Completions model and rejects requests
// that carry xAI server tools which are only usable through the Responses API.
type responsesAPIGatedModel struct {
	provider.LanguageModel
}

// Capabilities delegates to the wrapped model so the wrapper still satisfies
// provider.CapableModel (the embedded interface does not promote it).
func (m *responsesAPIGatedModel) Capabilities() provider.ModelCapabilities {
	if c, ok := m.LanguageModel.(provider.CapableModel); ok {
		return c.Capabilities()
	}
	return provider.ModelCapabilities{}
}

// DoGenerate rejects Responses-API-only tools before delegating to the
// underlying Chat Completions model.
func (m *responsesAPIGatedModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	if err := rejectResponsesOnlyTools(params.Tools); err != nil {
		return nil, err
	}
	return m.LanguageModel.DoGenerate(ctx, params)
}

// DoStream rejects Responses-API-only tools before delegating to the
// underlying Chat Completions model.
func (m *responsesAPIGatedModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if err := rejectResponsesOnlyTools(params.Tools); err != nil {
		return nil, err
	}
	return m.LanguageModel.DoStream(ctx, params)
}

// responsesOnlyTypes are xAI server tool types that only exist on the
// Responses API and cannot be expressed on the Chat Completions wire format.
var responsesOnlyTypes = map[string]bool{
	"web_search": true,
	"x_search":   true,
}

// rejectResponsesOnlyTools returns a descriptive error if any tool in tools is
// an xAI server tool that requires the Responses API.
func rejectResponsesOnlyTools(tools []provider.ToolDefinition) error {
	for _, t := range tools {
		if responsesOnlyTypes[t.ProviderDefinedType] {
			return fmt.Errorf(
				"xai: tool %q (%s) is only supported by xAI's Responses API; the goai xai provider uses Chat Completions, which does not support it",
				t.Name, t.ProviderDefinedType)
		}
	}
	return nil
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true, Image: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
