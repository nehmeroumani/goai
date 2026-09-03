package provider

import (
	"testing"
)

// --- NormalizeToolMessages ---

func TestNormalizeToolMessages_NoToolCalls(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "hello"}}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "hi"}}},
	}
	result := NormalizeToolMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if result[0].Role != RoleUser || result[1].Role != RoleAssistant {
		t.Errorf("roles = %s/%s, want user/assistant", result[0].Role, result[1].Role)
	}
}

func TestNormalizeToolMessages_MatchedPairs(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "weather?"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "weather"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "sunny"},
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "It is sunny"}}},
	}
	result := NormalizeToolMessages(msgs)
	// Should merge tool+assistant but no synthetic results needed.
	// Verify no "Tool execution aborted" anywhere.
	for _, m := range result {
		for _, p := range m.Content {
			if p.ToolOutput == "Tool execution aborted" {
				t.Error("unexpected synthetic tool result for matched pair")
			}
		}
	}
}

func TestNormalizeToolMessages_OrphanedToolCall(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "do stuff"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "do_thing"},
		}},
		// No tool result follows - orphan.
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "never mind"}}},
	}
	result := ensureToolResultPairing(msgs)
	// The synthetic result gets its own tool message between the assistant
	// turn and the user turn; the user message is left untouched.
	if len(result) != 4 {
		t.Fatalf("len = %d, want 4 (user, assistant, injected tool, user)", len(result))
	}
	injected := result[2]
	if injected.Role != RoleTool || len(injected.Content) != 1 {
		t.Fatalf("injected message = %+v, want one tool-result on a tool message", injected)
	}
	if p := injected.Content[0]; p.Type != PartToolResult || p.ToolCallID != "tc1" || p.ToolOutput != "Tool execution aborted" {
		t.Errorf("injected part = %+v", p)
	}
	if user := result[3]; user.Role != RoleUser || len(user.Content) != 1 || user.Content[0].Text != "never mind" {
		t.Errorf("user message = %+v, want the original text only", user)
	}
}

func TestNormalizeToolMessages_OrphanBeforeUserTurnKeepsToolRole(t *testing.T) {
	// End to end: the injected tool message and the following user turn merge
	// into one message that keeps the tool role, so providers that read tool
	// results only from tool messages still see the synthetic result.
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "do_thing"},
		}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "never mind"}}},
	}
	result := NormalizeToolMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2 (assistant, merged tool)", len(result))
	}
	merged := result[1]
	if merged.Role != RoleTool {
		t.Errorf("merged role = %s, want tool", merged.Role)
	}
	if len(merged.Content) != 2 || merged.Content[0].Type != PartToolResult || merged.Content[1].Text != "never mind" {
		t.Errorf("merged content = %+v, want [tool-result, text]", merged.Content)
	}
}

func TestNormalizeToolMessages_PartialMatch(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "multi"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "a"},
			{Type: PartToolCall, ToolCallID: "tc2", ToolName: "b"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "result-a"},
			// tc2 is orphaned
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "done"}}},
	}
	result := ensureToolResultPairing(msgs)
	// tc2's synthetic result is appended to the existing tool message, after
	// tc1's real result, rather than inserted as a new message.
	if len(result) != 4 {
		t.Fatalf("len = %d, want 4 (no message inserted)", len(result))
	}
	toolMsg := result[2]
	if toolMsg.Role != RoleTool || len(toolMsg.Content) != 2 {
		t.Fatalf("tool message = %+v, want two tool-results", toolMsg)
	}
	if p := toolMsg.Content[1]; p.ToolCallID != "tc2" || p.ToolOutput != "Tool execution aborted" {
		t.Errorf("appended part = %+v, want synthetic result for tc2", p)
	}
}

func TestNormalizeToolMessages_EndOfConversationOrphan(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "go"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "action"},
		}},
		// End of conversation - no following message at all.
	}
	result := ensureToolResultPairing(msgs)
	// Should insert a new tool message after assistant.
	if len(result) != 3 {
		t.Fatalf("len = %d, want 3 (user, assistant, injected tool)", len(result))
	}
	injected := result[2]
	if injected.Role != RoleTool {
		t.Errorf("injected message role = %s, want tool", injected.Role)
	}
	if len(injected.Content) != 1 {
		t.Fatalf("injected content len = %d, want 1", len(injected.Content))
	}
	p := injected.Content[0]
	if p.Type != PartToolResult || p.ToolCallID != "tc1" || p.ToolOutput != "Tool execution aborted" {
		t.Errorf("injected part = %+v", p)
	}
}

func TestEnsureToolResultPairing_SyntheticOrphanCarriesToolName(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "lookup_weather"},
		}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "cancelled"}}},
	}

	out := ensureToolResultPairing(msgs)

	var synthetic *Part
	for _, m := range out {
		for i := range m.Content {
			p := &m.Content[i]
			if p.Type == PartToolResult && p.ToolCallID == "tc1" && p.ToolOutput == "Tool execution aborted" {
				synthetic = p
			}
		}
	}
	if synthetic == nil {
		t.Fatal("expected synthetic tool result for orphaned tc1")
	}
	if synthetic.ToolName != "lookup_weather" {
		t.Errorf("ToolName = %q, want lookup_weather", synthetic.ToolName)
	}
}

func TestNormalizeToolMessages_MultipleConsecutiveToolMessages(t *testing.T) {
	// GoAI buildToolMessages pattern: multiple tool messages after one assistant.
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "run tools"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "a"},
			{Type: PartToolCall, ToolCallID: "tc2", ToolName: "b"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "r1"},
		}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc2", ToolOutput: "r2"},
		}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "done"}}},
	}
	result := NormalizeToolMessages(msgs)
	// Both tool results are matched - no synthetic results.
	for _, m := range result {
		for _, p := range m.Content {
			if p.ToolOutput == "Tool execution aborted" {
				t.Error("unexpected synthetic result - both tool calls are matched")
			}
		}
	}
}

// --- mergeConsecutiveRoles ---

func TestMergeConsecutiveRoles_SameRole(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "a"}}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "b"}}},
		{Role: RoleAssistant, Content: []Part{{Type: PartText, Text: "c"}}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	if len(result[0].Content) != 2 {
		t.Fatalf("merged content len = %d, want 2", len(result[0].Content))
	}
	if result[0].Content[0].Text != "a" || result[0].Content[1].Text != "b" {
		t.Errorf("merged texts = %q, %q", result[0].Content[0].Text, result[0].Content[1].Text)
	}
}

func TestMergeConsecutiveRoles_ToolAndUser(t *testing.T) {
	// Tool + user should merge (tool treated as user), with tool-result parts first.
	msgs := []Message{
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "result"},
		}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "thanks"}}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1 (merged)", len(result))
	}
	if result[0].Role != RoleTool {
		t.Errorf("merged role = %s, want tool", result[0].Role)
	}
	// Tool-result parts should come before text parts.
	if result[0].Content[0].Type != PartToolResult {
		t.Errorf("first part type = %s, want tool-result", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartText {
		t.Errorf("second part type = %s, want text", result[0].Content[1].Type)
	}
}

func TestMergeConsecutiveRoles_UserThenToolKeepsToolRole(t *testing.T) {
	// A user turn followed by a tool message merges the other way round, but
	// the tool-result still decides the role and still comes first.
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "note"}}},
		{Role: RoleTool, Content: []Part{
			{Type: PartToolResult, ToolCallID: "tc1", ToolOutput: "result"},
		}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 1 {
		t.Fatalf("len = %d, want 1 (merged)", len(result))
	}
	if result[0].Role != RoleTool {
		t.Errorf("merged role = %s, want tool", result[0].Role)
	}
	if result[0].Content[0].Type != PartToolResult || result[0].Content[1].Text != "note" {
		t.Errorf("merged content = %+v, want [tool-result, text]", result[0].Content)
	}
}

func TestMergeConsecutiveRoles_UserTextOnlyKeepsUserRole(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "one"}}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "two"}}},
	}
	result := mergeConsecutiveRoles(msgs)
	if len(result) != 1 || result[0].Role != RoleUser || len(result[0].Content) != 2 {
		t.Fatalf("result = %+v, want one user message with two parts", result)
	}
}

func TestMergeConsecutiveRoles_Empty(t *testing.T) {
	result := mergeConsecutiveRoles(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}

// --- ReorderAssistantParts ---

func TestReorderAssistantParts_TextBeforeToolCall(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "fn"},
			{Type: PartText, Text: "thinking..."},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if result[0].Content[0].Type != PartText {
		t.Errorf("first part = %s, want text", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartToolCall {
		t.Errorf("second part = %s, want tool-call", result[0].Content[1].Type)
	}
}

func TestReorderAssistantParts_AlreadyOrdered(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []Part{
			{Type: PartText, Text: "let me check"},
			{Type: PartToolCall, ToolCallID: "tc1", ToolName: "fn"},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if result[0].Content[0].Type != PartText {
		t.Errorf("first part = %s, want text (unchanged)", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != PartToolCall {
		t.Errorf("second part = %s, want tool-call (unchanged)", result[0].Content[1].Type)
	}
}

func TestReorderAssistantParts_NonAssistantUntouched(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{
			{Type: PartText, Text: "hello"},
		}},
	}
	result := ReorderAssistantParts(msgs)
	if len(result) != 1 || result[0].Content[0].Text != "hello" {
		t.Error("user message should be untouched")
	}
}

func TestReorderAssistantParts_Empty(t *testing.T) {
	result := ReorderAssistantParts(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}

func TestNormalizeToolMessages_EmptyMessages(t *testing.T) {
	result := NormalizeToolMessages(nil)
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
	result = NormalizeToolMessages([]Message{})
	if len(result) != 0 {
		t.Fatalf("len = %d, want 0", len(result))
	}
}

// TestEnsureToolResultPairing_SkipsServerExecuted verifies that tool calls
// whose ProviderOptions["resultBlock"] is set (server-executed) are NOT
// considered orphaned: their result is delivered inline on the assistant
// turn, so no synthetic tool_result message is injected.
func TestEnsureToolResultPairing_SkipsServerExecuted(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "search go"}}},
		{Role: RoleAssistant, Content: []Part{
			{Type: PartText, Text: "Searching..."},
			{
				Type:       PartToolCall,
				ToolCallID: "srvtoolu_1",
				ToolName:   "web_search",
				ProviderOptions: map[string]any{
					"resultBlock": map[string]any{"type": "web_search_tool_result"},
				},
			},
		}},
		{Role: RoleUser, Content: []Part{{Type: PartText, Text: "thanks"}}},
	}

	out := ensureToolResultPairing(msgs)

	// No synthetic tool message must be inserted between the assistant and
	// the next user message -- the original three messages must be intact.
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3 (no orphan injected for server tool)", len(out))
	}
	if out[1].Role != RoleAssistant {
		t.Errorf("msg[1].Role = %v, want assistant", out[1].Role)
	}
	if out[2].Role != RoleUser {
		t.Errorf("msg[2].Role = %v, want user (not synthetic tool)", out[2].Role)
	}
}
