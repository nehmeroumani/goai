package openaicompat

// BoolPtr returns a pointer to b. Used for config flags that distinguish an
// explicit false from an unset value. Hoisted here so provider wrappers share
// a single implementation instead of duplicating it.
func BoolPtr(b bool) *bool { return &b }

// MergeHeaders returns a new header map containing fixed headers overlaid by
// user headers (user wins). It never mutates its inputs. Hoisted here so
// provider wrappers (OpenRouter, Requesty, ...) share one implementation
// instead of duplicating it.
func MergeHeaders(user, fixed map[string]string) map[string]string {
	merged := make(map[string]string, len(fixed)+len(user))
	for k, v := range fixed {
		merged[k] = v
	}
	for k, v := range user {
		merged[k] = v
	}
	return merged
}
