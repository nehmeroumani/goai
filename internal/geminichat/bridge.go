package geminichat

import (
	"net/http"

	"github.com/zendev-sh/goai/provider"
)

// Config contains transport-neutral native Gemini chat configuration.
type Config struct {
	Project     string
	Location    string
	BaseURL     string
	TokenSource provider.TokenSource
	Headers     map[string]string
	HTTPClient  *http.Client
}

var factory func(string, Config) provider.LanguageModel

// RegisterFactory installs the native Gemini model implementation.
func RegisterFactory(f func(string, Config) provider.LanguageModel) {
	factory = f
}

// New creates a native Gemini language model through the registered provider.
func New(modelID string, cfg Config) provider.LanguageModel {
	if factory == nil {
		panic("geminichat: provider factory is not registered")
	}
	return factory(modelID, cfg)
}
