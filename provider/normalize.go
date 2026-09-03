package provider

import "slices"

// NormalizeOption adjusts how NormalizeToolMessages shapes its output.
type NormalizeOption func(*normalizeConfig)

type normalizeConfig struct {
	separateToolMessages bool
}

// WithSeparateToolMessages keeps tool messages as turns of their own instead
// of folding them into adjacent user messages. Use it before providers with a
// dedicated tool role (OpenAI-compatible, Cohere), where a tool message
// followed by a user message is a valid sequence and merging the two would
// leave user text inside a tool message.
func WithSeparateToolMessages() NormalizeOption {
	return func(c *normalizeConfig) { c.separateToolMessages = true }
}

// NormalizeToolMessages prepares messages for providers that require:
// 1. Every assistant tool-call has a matching tool-result (orphan fix)
// 2. Alternating user/assistant roles (merge consecutive same-role)
//
// Tool-result parts only ever live on RoleTool messages in the output: a
// synthetic result gets its own tool message, and a merge that absorbs a
// tool result keeps the tool role. Providers whose wire format carries tool
// results inside a user turn (Anthropic, Bedrock) map the role at conversion
// time. Providers with a dedicated tool role (OpenAI-compatible, Cohere) read
// tool results only from tool messages, so this invariant is what keeps the
// synthetic results visible to them; pass WithSeparateToolMessages for those
// providers so tool and user messages are not merged at all.
//
// Call this before provider-specific message conversion.
func NormalizeToolMessages(msgs []Message, opts ...NormalizeOption) []Message {
	var cfg normalizeConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	msgs = ensureToolResultPairing(msgs)
	msgs = mergeConsecutiveRoles(msgs, !cfg.separateToolMessages)
	return msgs
}

// ensureToolResultPairing ensures every assistant message with tool-call parts
// has matching tool-result parts in following messages. Injects synthetic
// "Tool execution aborted" results for orphaned tool-calls, appended to the
// tool message that directly follows the assistant turn, or placed on a new
// tool message inserted right after it. Results never land on a user message.
//
// Server-executed tool calls (e.g. Anthropic web_search) are skipped: their
// result is delivered inline on the same assistant turn via the part's
// ProviderOptions["resultBlock"], so no following tool-result message is
// expected.
func ensureToolResultPairing(msgs []Message) []Message {
	msgs = cloneMessages(msgs)
	for i := 0; i < len(msgs); i++ {
		if msgs[i].Role != RoleAssistant {
			continue
		}
		var toolCalls []Part
		for _, p := range msgs[i].Content {
			if p.Type == PartToolCall && p.ToolCallID != "" {
				if _, hasInlineResult := p.ProviderOptions["resultBlock"]; hasInlineResult {
					continue
				}
				toolCalls = append(toolCalls, p)
			}
		}
		if len(toolCalls) == 0 {
			continue
		}
		// Scan all consecutive tool/user messages after assistant
		resultIDs := make(map[string]bool)
		for j := i + 1; j < len(msgs); j++ {
			r := msgs[j].Role
			if r != RoleTool && r != RoleUser {
				break
			}
			for _, p := range msgs[j].Content {
				if p.Type == PartToolResult && p.ToolCallID != "" {
					resultIDs[p.ToolCallID] = true
				}
			}
		}
		var orphans []Part
		for _, toolCall := range toolCalls {
			if !resultIDs[toolCall.ToolCallID] {
				orphans = append(orphans, Part{
					Type:       PartToolResult,
					ToolCallID: toolCall.ToolCallID,
					ToolOutput: "Tool execution aborted",
					ToolName:   toolCall.ToolName,
				})
			}
		}
		if len(orphans) == 0 {
			continue
		}
		if i+1 < len(msgs) && msgs[i+1].Role == RoleTool {
			msgs[i+1].Content = append(msgs[i+1].Content, orphans...)
		} else {
			msgs = slices.Insert(msgs, i+1, Message{Role: RoleTool, Content: orphans})
		}
	}
	return msgs
}

func cloneMessages(msgs []Message) []Message {
	if len(msgs) == 0 {
		return msgs
	}
	out := make([]Message, len(msgs))
	for i, msg := range msgs {
		// Copy the whole struct (preserving ProviderOptions and any future
		// fields), then clone the Content slice, which is the only part this
		// package appends to.
		out[i] = msg
		out[i].Content = slices.Clone(msg.Content)
	}
	return out
}

// mergeConsecutiveRoles merges consecutive messages with the same role. When
// foldToolIntoUser is set, tool-role messages count as user-role for that
// comparison, and a merged message that carries tool-result parts keeps the
// tool role and places those parts first (providers require tool-result
// immediately after tool-use). Otherwise tool messages only merge with other
// tool messages.
func mergeConsecutiveRoles(msgs []Message, foldToolIntoUser bool) []Message {
	if len(msgs) == 0 {
		return msgs
	}
	var result []Message
	for _, msg := range msgs {
		if len(result) > 0 && mergeRole(result[len(result)-1].Role, foldToolIntoUser) == mergeRole(msg.Role, foldToolIntoUser) {
			last := &result[len(result)-1]
			var toolResults, others []Part
			for _, p := range slices.Concat(last.Content, msg.Content) {
				if p.Type == PartToolResult {
					toolResults = append(toolResults, p)
				} else {
					others = append(others, p)
				}
			}
			if len(toolResults) > 0 {
				last.Role = RoleTool
			}
			last.Content = append(toolResults, others...)
			continue
		}
		result = append(result, msg)
	}
	return result
}

// mergeRole is the role used to detect consecutive same-role messages: the
// tool role folds into user when foldToolIntoUser is set.
func mergeRole(role Role, foldToolIntoUser bool) Role {
	if foldToolIntoUser && role == RoleTool {
		return RoleUser
	}
	return role
}

// ReorderAssistantParts sorts assistant message parts so text/reasoning
// come before tool-call parts. Anthropic/Bedrock require this ordering.
func ReorderAssistantParts(msgs []Message) []Message {
	for i := range msgs {
		if msgs[i].Role != RoleAssistant {
			continue
		}
		var textParts, toolCallParts []Part
		for _, p := range msgs[i].Content {
			if p.Type == PartToolCall {
				toolCallParts = append(toolCallParts, p)
			} else {
				textParts = append(textParts, p)
			}
		}
		if len(toolCallParts) > 0 && len(textParts) > 0 {
			msgs[i].Content = append(textParts, toolCallParts...)
		}
	}
	return msgs
}
