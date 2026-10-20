package tool

import (
	"context"

	"github.com/google/uuid"
)

type searchSessionKey struct{}
type searchResolverKey struct{}

// WithSearchSession attaches an opaque standalone research-run ID.
func WithSearchSession(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, searchSessionKey{}, id)
}

// WithSearchSessionResolver defers conversation-store access until search runs.
// Other tools, explicit replacements and disabled search never invoke it.
func WithSearchSessionResolver(ctx context.Context, resolve func(context.Context) (string, error)) context.Context {
	return context.WithValue(ctx, searchResolverKey{}, resolve)
}

// EnsureSearchSession starts one standalone research scope only when the caller
// has not already supplied a conversation scope. It never accesses storage.
func EnsureSearchSession(ctx context.Context) context.Context {
	if _, ok := ctx.Value(searchResolverKey{}).(func(context.Context) (string, error)); ok {
		return ctx
	}
	if id, ok := ctx.Value(searchSessionKey{}).(string); ok && id != "" {
		return ctx
	}
	return WithSearchSession(ctx, uuid.NewString())
}

// SearchSession resolves the current conversation ID. A persistence failure is
// returned to the search tool; it never rotates identity or silently falls back.
func SearchSession(ctx context.Context) (string, error) {
	if resolve, ok := ctx.Value(searchResolverKey{}).(func(context.Context) (string, error)); ok {
		return resolve(ctx)
	}
	if id, ok := ctx.Value(searchSessionKey{}).(string); ok && id != "" {
		return id, nil
	}
	return uuid.NewString(), nil
}
