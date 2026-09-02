package geminichat

import (
	"context"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

type testModel struct{ id string }

func (m *testModel) ModelID() string { return m.id }

func (m *testModel) DoGenerate(context.Context, provider.GenerateParams) (*provider.GenerateResult, error) {
	return &provider.GenerateResult{}, nil
}

func (m *testModel) DoStream(context.Context, provider.GenerateParams) (*provider.StreamResult, error) {
	return &provider.StreamResult{}, nil
}

func TestFactoryRegistration(t *testing.T) {
	original := factory
	t.Cleanup(func() { factory = original })

	factory = nil
	func() {
		defer func() {
			if recover() == nil {
				t.Error("New() should panic without a registered factory")
			}
		}()
		_ = New("missing", Config{})
	}()

	RegisterFactory(func(modelID string, _ Config) provider.LanguageModel {
		return &testModel{id: modelID}
	})
	if got := New("registered", Config{}).ModelID(); got != "registered" {
		t.Errorf("ModelID() = %q, want registered", got)
	}
}
