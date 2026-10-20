package tool

import (
	"context"

	"github.com/google/uuid"
)

type searchSessionKey struct{}

// WithSearchSession attaches an opaque conversation ID, independent of MCP
// connections. Callers must reuse it for follow-up turns in that conversation.
func WithSearchSession(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, searchSessionKey{}, id)
}

// SearchSession returns the conversation ID, or starts a standalone research run.
func SearchSession(ctx context.Context) string {
	if id, ok := ctx.Value(searchSessionKey{}).(string); ok && id != "" {
		return id
	}
	return uuid.NewString()
}
