package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/enterprise/ai-agent-go/internal/tool"
)

func TestSearchMCPDispatch(t *testing.T) {
	for _, sse := range []bool{false, true} {
		t.Run(fmt.Sprint("sse=", sse), func(t *testing.T) {
			var methods []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.UserAgent() != parallelUserAgent || r.Header.Get("Authorization") != "" {
					t.Error("wrong user agent or ambient authentication")
				}
				if r.Method == http.MethodDelete {
					w.WriteHeader(http.StatusNoContent)
					return
				}
				if r.Method != http.MethodPost || r.Header.Get("Accept") != "application/json, text/event-stream" {
					t.Error("wrong transport headers")
				}
				var req struct {
					ID     int             `json:"id"`
					Method string          `json:"method"`
					Params json.RawMessage `json:"params"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				methods = append(methods, req.Method)
				var result interface{}
				switch req.Method {
				case "initialize":
					result = map[string]interface{}{"protocolVersion": "2025-03-26", "capabilities": map[string]interface{}{}, "serverInfo": map[string]string{"name": "fixture", "version": "1"}}
					w.Header().Set("Mcp-Session-Id", "http-session")
				case "notifications/initialized":
					w.WriteHeader(http.StatusAccepted)
					return
				case "tools/list":
					result = map[string]interface{}{"tools": []map[string]string{{"name": "web_search"}}}
				case "tools/call":
					if r.Header.Get("Mcp-Session-Id") != "http-session" {
						t.Error("lost HTTP session")
					}
					if r.Header.Get("MCP-Protocol-Version") != "2025-03-26" {
						t.Error("wrong protocol")
					}
					var params struct {
						Name      string                 `json:"name"`
						Arguments map[string]interface{} `json:"arguments"`
					}
					json.Unmarshal(req.Params, &params)
					if params.Name != "web_search" || params.Arguments["session_id"] != "opaque-conversation" || params.Arguments["objective"] != "Go release" {
						t.Errorf("wrong args: %s", req.Params)
					}
					queries, ok := params.Arguments["search_queries"].([]interface{})
					if !ok || len(queries) != 1 || queries[0] != "Go release" {
						t.Error("wrong queries")
					}
					if _, ok := params.Arguments["max_results"]; ok {
						t.Error("unsupported server argument")
					}
					result = map[string]interface{}{"content": []map[string]string{{"type": "text", "text": `{"results":[{"url":"https://go.dev/doc/","title":"Go","excerpts":["release notes"]},{"url":"https://go.dev/other"}]}`}}}
				default:
					t.Errorf("unexpected method %s", req.Method)
				}
				data, _ := json.Marshal(map[string]interface{}{"jsonrpc": "2.0", "id": req.ID, "result": result})
				if sse {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprintf(w, ": keepalive\n\nevent: message\ndata: %s\n\n", data)
				} else {
					w.Header().Set("Content-Type", "application/json")
					w.Write(data)
				}
			}))
			defer server.Close()
			search := NewSearchTool(zap.NewNop())
			search.endpoint = server.URL
			registry := tool.NewRegistry()
			registry.MustRegister(search)
			router := tool.NewRouter(registry, zap.NewNop())
			result, err := router.Execute(tool.WithSearchSession(context.Background(), "opaque-conversation"), "web_search", `{"query":" Go release ","max_results":1}`)
			if err != nil {
				t.Fatal(err)
			}
			if !result.Success || !strings.Contains(result.Output, "https://go.dev/doc/") || !strings.Contains(result.Output, "release notes") || strings.Contains(result.Output, "https://go.dev/other") {
				t.Fatalf("bad output: %+v", result)
			}
			if strings.Join(methods, ",") != "initialize,notifications/initialized,tools/list,tools/call" {
				t.Fatal(methods)
			}
			registry.Unregister("web_search")
			if _, err := router.Execute(context.Background(), "web_search", `{"query":"blocked"}`); err == nil {
				t.Fatal("unregistered tool executed")
			}
		})
	}
}

func TestSearchInvalidArguments(t *testing.T) {
	search := NewSearchTool(zap.NewNop())
	for _, input := range []string{`{`, `{}`, `{"query":"  "}`, `{"query":"test","max_results":21}`} {
		result, err := search.Execute(context.Background(), input)
		if err != nil || result == nil || result.Success {
			t.Fatalf("invalid input accepted: %s, %+v %v", input, result, err)
		}
	}
}

func TestMCPFailures(t *testing.T) {
	for _, tc := range []struct {
		name        string
		status      int
		media, body string
	}{
		{"http", 429, "application/json", `quota exceeded`},
		{"rpc", 200, "application/json", `{"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"unavailable"}}`},
		{"malformed", 200, "application/json", `{`},
		{"wrong id", 200, "application/json", `{"jsonrpc":"2.0","id":2,"result":{}}`},
		{"missing result", 200, "application/json", `{"jsonrpc":"2.0","id":1}`},
		{"media", 200, "text/plain", `hello`},
		{"stream end", 200, "text/event-stream", ": keepalive\n\n"},
		{"oversize", 200, "application/json", strings.Repeat("x", maxMCPResponse+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.media)
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			search := NewSearchTool(zap.NewNop())
			search.endpoint = server.URL
			if _, err := search.Execute(context.Background(), `{"query":"test"}`); err == nil {
				t.Fatal("failure reported as success")
			}
		})
	}
}

func TestMCPCancellationAndTimeout(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		t.Run(fmt.Sprint(timeout), func(t *testing.T) {
			started := make(chan struct{})
			var once sync.Once
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				once.Do(func() { close(started) })
				<-r.Context().Done()
			}))
			defer server.Close()
			search := NewSearchTool(zap.NewNop())
			search.endpoint = server.URL
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if timeout {
				var stop context.CancelFunc
				ctx, stop = context.WithTimeout(ctx, 30*time.Millisecond)
				defer stop()
			} else {
				go func() { <-started; cancel() }()
			}
			before := time.Now()
			_, err := search.Execute(ctx, `{"query":"test"}`)
			if err == nil || time.Since(before) > time.Second {
				t.Fatalf("cancellation/timeout failed: %v", err)
			}
		})
	}
}

// Opt in to a real anonymous call; the regular suite is deterministic and offline.
func TestParallelLiveSearch(t *testing.T) {
	if os.Getenv("RUN_PARALLEL_LIVE") != "1" {
		t.Skip("set RUN_PARALLEL_LIVE=1 for an anonymous network call")
	}
	registry := tool.NewRegistry()
	registry.MustRegister(NewSearchTool(zap.NewNop()))
	result, err := tool.NewRouter(registry, zap.NewNop()).Execute(context.Background(), "web_search", `{"query":"Go programming language official documentation","max_results":2}`)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Success || !strings.Contains(result.Output, "https://") {
		t.Fatalf("missing useful URLs: %+v", result)
	}
	t.Log(result.Output)
}

func TestMCPToolFailures(t *testing.T) {
	for _, payload := range []string{
		`{"isError":true,"content":[{"type":"text","text":"search unavailable"}]}`,
		`{"content":[{"type":"text","text":"not JSON"}]}`,
		`{"content":[{"type":"text","text":"{}"}]}`,
		`{"content":[{"type":"image"}]}`,
	} {
		t.Run(payload, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					ID     int    `json:"id"`
					Method string `json:"method"`
				}
				json.NewDecoder(r.Body).Decode(&req)
				if req.ID == 0 {
					w.WriteHeader(202)
					return
				}
				result := `{"protocolVersion":"2025-03-26"}`
				if req.Method == "tools/list" {
					result = `{"tools":[{"name":"web_search"}]}`
				}
				if req.Method == "tools/call" {
					result = payload
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":%s}`, req.ID, result)
			}))
			defer server.Close()
			search := NewSearchTool(zap.NewNop())
			search.endpoint = server.URL
			registry := tool.NewRegistry()
			registry.MustRegister(search)
			if _, err := tool.NewRouter(registry, zap.NewNop()).Execute(context.Background(), "web_search", `{"query":"test"}`); err == nil {
				t.Fatal("invalid tool response treated as success")
			}
		})
	}
}

type explicitSearch struct{}

func (explicitSearch) Name() string                       { return "web_search" }
func (explicitSearch) Description() string                { return "Explicit user replacement" }
func (explicitSearch) Parameters() map[string]interface{} { return map[string]interface{}{} }
func (explicitSearch) Execute(context.Context, string) (*tool.ToolResult, error) {
	return tool.NewSuccessResult("explicit provider selected"), nil
}

func TestSearchRegistryOverride(t *testing.T) {
	registry := tool.NewRegistry()
	registry.MustRegister(explicitSearch{})
	// Existing explicit registrations are never overwritten by default setup.
	if err := registry.Register(NewSearchTool(zap.NewNop())); err == nil {
		t.Fatal("overwrote an explicit provider")
	}
	result, err := tool.NewRouter(registry, zap.NewNop()).Execute(context.Background(), "web_search", `{"query":"test"}`)
	if err != nil || result.Output != "explicit provider selected" {
		t.Fatalf("lost explicit provider: %+v %v", result, err)
	}
}

// A server-assigned HTTP session must not make cancellation wait for cleanup.
func TestMCPCancelAfterInitialize(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(fmt.Sprint("deadline=", deadline), func(t *testing.T) {
			reachedSearch := make(chan struct{})
			release := make(chan struct{})
			var once sync.Once
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodDelete {
					<-release
					w.WriteHeader(204)
					return
				}
				var req struct {
					ID     int    `json:"id"`
					Method string `json:"method"`
				}
				json.NewDecoder(r.Body).Decode(&req)
				w.Header().Set("Content-Type", "application/json")
				switch req.Method {
				case "initialize":
					w.Header().Set("Mcp-Session-Id", "assigned")
					fmt.Fprint(w, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}`)
				case "notifications/initialized":
					w.WriteHeader(202)
				case "tools/list":
					fmt.Fprint(w, `{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"web_search"}]}}`)
				case "tools/call":
					once.Do(func() { close(reachedSearch) })
					<-release
				}
			}))
			defer server.Close()
			defer close(release)
			search := NewSearchTool(zap.NewNop())
			search.endpoint = server.URL
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if deadline {
				var stop context.CancelFunc
				ctx, stop = context.WithTimeout(ctx, 100*time.Millisecond)
				defer stop()
			} else {
				go func() { <-reachedSearch; cancel() }()
			}
			start := time.Now()
			_, err := search.Execute(ctx, `{"query":"test"}`)
			if err == nil {
				t.Fatal("canceled search succeeded")
			}
			if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
				t.Fatalf("canceled search waited for session cleanup: %v", elapsed)
			}
		})
	}
}
