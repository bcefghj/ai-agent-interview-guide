package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/enterprise/ai-agent-go/internal/config"
	"github.com/enterprise/ai-agent-go/internal/intent"
	"github.com/enterprise/ai-agent-go/internal/llm"
	"github.com/enterprise/ai-agent-go/internal/memory"
	"github.com/enterprise/ai-agent-go/internal/model"
	"github.com/enterprise/ai-agent-go/internal/tool"
	"github.com/enterprise/ai-agent-go/internal/tool/builtin"
)

// Scripted model, not live inference: request an actual registered search and
// return a final answer only after its source URL reaches the next model turn.
type searchModel struct{ live bool }

func (*searchModel) Name() string                 { return "scripted" }
func (*searchModel) Healthy(context.Context) bool { return true }
func (*searchModel) ChatStream(context.Context, *model.LLMRequest) (<-chan llm.StreamEvent, error) {
	return nil, fmt.Errorf("unused")
}
func (m *searchModel) Chat(_ context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	if len(req.Messages) == 1 && strings.Contains(req.Messages[0].Content, "意图识别引擎") {
		return &model.LLMResponse{Content: `{"intent":"tool_use","confidence":1}`}, nil
	}
	last := req.Messages[len(req.Messages)-1].Content
	if strings.HasPrefix(last, "Observation:") {
		if strings.Contains(last, "MCP HTTP 429") {
			return &model.LLMResponse{Content: "Final Answer: Search unavailable"}, nil
		}
		if m.live {
			var payload struct {
				Results []struct {
					URL      string   `json:"url"`
					Excerpts []string `json:"excerpts"`
				} `json:"results"`
			}
			if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(last, "Observation:"))), &payload); err != nil || len(payload.Results) == 0 || !strings.HasPrefix(payload.Results[0].URL, "https://") || len(payload.Results[0].Excerpts) == 0 {
				return nil, fmt.Errorf("live search returned no useful source: %s", last)
			}
			return &model.LLMResponse{Content: "Final Answer: Source: " + payload.Results[0].URL}, nil
		}
		if !strings.Contains(last, "https://go.dev/doc/") || !strings.Contains(last, "excerpts") {
			return nil, fmt.Errorf("search output not consumed: %s", last)
		}
		return &model.LLMResponse{Content: "Final Answer: Current Go documentation: https://go.dev/doc/"}, nil
	}
	action, _ := json.Marshal(map[string]string{"tool": "web_search", "input": `{"query":"Go documentation","max_results":1}`})
	return &model.LLMResponse{Content: "Thought: Search for a current source.\nAction: " + string(action)}, nil
}

type emptyMemory struct{}

func (*emptyMemory) Save(context.Context, *model.MemoryEntry) error                  { return nil }
func (*emptyMemory) Load(context.Context, string, int) ([]*model.MemoryEntry, error) { return nil, nil }
func (*emptyMemory) Search(context.Context, string, int) ([]*model.MemoryEntry, error) {
	return nil, nil
}
func (*emptyMemory) Clear(context.Context, string) error { return nil }

type captureSearchTransport struct {
	t        *testing.T
	mu       sync.Mutex
	ids      []string
	failNext bool
}

func (c *captureSearchTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.String() != "https://search.parallel.ai/mcp" {
		c.t.Errorf("wrong destination: %s", r.URL)
	}
	if r.UserAgent() != "ai-agent-interview-guide/0.1 (+https://github.com/bcefghj/ai-agent-interview-guide)" {
		c.t.Error("wrong project User-Agent")
	}
	if r.Header.Get("Authorization") != "" {
		c.t.Error("ambient credentials used")
	}
	var req struct {
		ID     int    `json:"id"`
		Method string `json:"method"`
		Params struct {
			Arguments struct {
				Session   string   `json:"session_id"`
				Objective string   `json:"objective"`
				Queries   []string `json:"search_queries"`
			} `json:"arguments"`
		} `json:"params"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, err
	}
	var result interface{}
	switch req.Method {
	case "initialize":
		result = map[string]interface{}{"protocolVersion": "2025-03-26"}
	case "notifications/initialized":
		return &http.Response{StatusCode: 202, Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header)}, nil
	case "tools/list":
		result = map[string]interface{}{"tools": []map[string]string{{"name": "web_search"}}}
	case "tools/call":
		if _, err := uuid.Parse(req.Params.Arguments.Session); err != nil {
			c.t.Errorf("not an opaque UUID: %s", req.Params.Arguments.Session)
		}
		if req.Params.Arguments.Objective != "Go documentation" || len(req.Params.Arguments.Queries) != 1 || req.Params.Arguments.Queries[0] != "Go documentation" {
			c.t.Errorf("wrong search params: %+v", req.Params)
		}
		c.mu.Lock()
		c.ids = append(c.ids, req.Params.Arguments.Session)
		fail := c.failNext
		c.failNext = false
		c.mu.Unlock()
		if fail {
			return &http.Response{StatusCode: 429, Body: io.NopCloser(strings.NewReader("quota exceeded")), Header: make(http.Header)}, nil
		}
		result = map[string]interface{}{"content": []map[string]string{{"type": "text", "text": `{"results":[{"title":"Go","url":"https://go.dev/doc/","excerpts":["Current Go documentation"]}]}`}}}
	default:
		return nil, fmt.Errorf("unexpected method: %s", req.Method)
	}
	data, _ := json.Marshal(map[string]interface{}{"jsonrpc": "2.0", "id": req.ID, "result": result})
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(string(data)))}, nil
}

func TestSearchThroughOrchestrator(t *testing.T) {
	// Capture the production destination through net/http; do not substitute a
	// fake tool or call the MCP helper directly.
	capture := &captureSearchTransport{t: t}
	previous := http.DefaultTransport
	http.DefaultTransport = capture
	t.Cleanup(func() { http.DefaultTransport = previous })
	t.Setenv("PARALLEL_API_KEY", "ambient-must-not-be-used")
	logger := zap.NewNop()
	registry := tool.NewRegistry()
	registry.MustRegister(builtin.NewSearchTool(logger))
	modelRouter := llm.NewRouter(map[string]llm.Client{"scripted": &searchModel{}}, "scripted", config.CBConfig{FailureThreshold: 100, SuccessThreshold: 1})
	sessionCache := &conversationCache{values: make(map[string]string)}
	manager := memory.NewManager(memory.NewShortTermMemory(sessionCache, 20), &emptyMemory{})
	orchestrator := NewOrchestrator(OrchestratorDeps{ModelRouter: modelRouter, MemoryManager: manager, ToolRouter: tool.NewRouter(registry, logger), IntentRecognizer: intent.NewRecognizer(modelRouter, logger), Config: config.AgentConfig{MaxIterations: 3}, Logger: logger})
	run := func(session string) {
		response, err := orchestrator.ProcessMessage(context.Background(), &model.ChatRequest{SessionID: session, Message: "Find Go documentation"})
		if err != nil {
			t.Error(err)
			return
		}
		if response.Content != "Current Go documentation: https://go.dev/doc/" || len(response.ToolCalls) != 1 || !strings.Contains(response.ToolCalls[0].Output, "https://go.dev/doc/") {
			t.Errorf("bad agent response: %+v", response)
		}
	}
	run("user@example.com") // user-controlled conversation keys must not leave the app
	manager = memory.NewManager(memory.NewShortTermMemory(sessionCache, 20), &emptyMemory{})
	orchestrator.deps.MemoryManager = manager
	run("user@example.com")
	run("independent")
	var wg sync.WaitGroup
	for _, session := range []string{"user@example.com", "independent"} {
		wg.Add(1)
		go func(s string) { defer wg.Done(); run(s) }(session)
	}
	wg.Wait()
	capture.mu.Lock()
	ids := append([]string(nil), capture.ids...)
	capture.mu.Unlock()
	if len(ids) != 5 || ids[0] != ids[1] || ids[0] == ids[2] {
		t.Fatalf("wrong conversation scope: %v", ids)
	}
	for _, id := range ids[3:] {
		if id != ids[0] && id != ids[2] {
			t.Fatalf("interleaved identity changed: %v", ids)
		}
	}
	if err := manager.ClearSession(context.Background(), "user@example.com"); err != nil {
		t.Fatal(err)
	}
	run("user@example.com")
	capture.mu.Lock()
	resetID := capture.ids[len(capture.ids)-1]
	capture.mu.Unlock()
	if resetID == ids[0] {
		t.Fatal("reset reused identity")
	}
	// A quota response followed by a user retry keeps the conversation identity.
	capture.mu.Lock()
	capture.failNext = true
	capture.mu.Unlock()
	failed, failErr := orchestrator.ProcessMessage(context.Background(), &model.ChatRequest{SessionID: "user@example.com", Message: "Find Go documentation"})
	if failErr != nil || failed.Content != "Search unavailable" || !strings.Contains(failed.ToolCalls[0].Output, "MCP HTTP 429") {
		t.Fatalf("tool error was not consumed: %+v, %v", failed, failErr)
	}
	run("user@example.com")
	capture.mu.Lock()
	for _, id := range capture.ids[5:] {
		if id != resetID {
			t.Errorf("retry changed identity: %v", capture.ids)
		}
	}
	capture.mu.Unlock()
	// Concurrent first turns in one new conversation select a single persisted ID.
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); run("concurrent-first-turn") }()
	}
	wg.Wait()
	capture.mu.Lock()
	defer capture.mu.Unlock()
	if len(capture.ids) != 10 || capture.ids[8] != capture.ids[9] || capture.ids[8] == resetID {
		t.Fatalf("concurrent first creation split identities: %v", capture.ids)
	}
}

// Real MCP, scripted model. This verifies live results reach the agent consumer
// while keeping model inference deterministic and requiring no model API key.
func TestSearchThroughOrchestratorLive(t *testing.T) {
	if os.Getenv("RUN_PARALLEL_LIVE_AGENT") != "1" {
		t.Skip("set RUN_PARALLEL_LIVE_AGENT=1 for a live anonymous agent search")
	}
	previous := http.DefaultTransport
	observed := &observedLiveTransport{t: t, base: previous}
	http.DefaultTransport = observed
	t.Cleanup(func() { http.DefaultTransport = previous })
	t.Setenv("PARALLEL_API_KEY", "ambient-must-not-be-used")
	logger := zap.NewNop()
	registry := tool.NewRegistry()
	registry.MustRegister(builtin.NewSearchTool(logger))
	modelRouter := llm.NewRouter(map[string]llm.Client{"scripted": &searchModel{live: true}}, "scripted", config.CBConfig{FailureThreshold: 100, SuccessThreshold: 1})
	sessionCache := &conversationCache{values: make(map[string]string)}
	manager := memory.NewManager(memory.NewShortTermMemory(sessionCache, 20), &emptyMemory{})
	orchestrator := NewOrchestrator(OrchestratorDeps{ModelRouter: modelRouter, MemoryManager: manager, ToolRouter: tool.NewRouter(registry, logger), IntentRecognizer: intent.NewRecognizer(modelRouter, logger), Config: config.AgentConfig{MaxIterations: 3}, Logger: logger})
	result, err := orchestrator.ProcessMessage(context.Background(), &model.ChatRequest{SessionID: "local-live-conversation", Message: "Find Go documentation"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || !strings.HasPrefix(result.Content, "Source: https://") {
		t.Fatalf("search did not reach model consumer: %+v", result)
	}
	if observed.requests < 4 {
		t.Fatalf("missing discovery/search requests: %d", observed.requests)
	}
	t.Logf("Live MCP result consumed by scripted ReAct model: %s", result.Content)
}

type observedLiveTransport struct {
	t        *testing.T
	base     http.RoundTripper
	requests int
}

func (o *observedLiveTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.String() != "https://search.parallel.ai/mcp" || r.UserAgent() != "ai-agent-interview-guide/0.1 (+https://github.com/bcefghj/ai-agent-interview-guide)" || r.Header.Get("Authorization") != "" {
		o.t.Fatalf("wrong live request destination, User-Agent or authentication")
	}
	if r.Method == http.MethodDelete {
		o.t.Logf("Observed live request: DELETE %s; User-Agent=%s; Authorization absent", r.URL, r.UserAgent())
		o.requests++
		return o.base.RoundTrip(r)
	}
	data, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	r.Body = io.NopCloser(strings.NewReader(string(data)))
	var request struct {
		Method string `json:"method"`
	}
	if err := json.Unmarshal(data, &request); err != nil {
		return nil, err
	}
	o.t.Logf("Observed live request: %s %s; method=%s; User-Agent=%s; Authorization absent", r.Method, r.URL, request.Method, r.UserAgent())
	o.requests++
	return o.base.RoundTrip(r)
}

// Deterministic Redis substitute. Identity creation still executes the production
// ShortTermMemory Get/SET NX path, including concurrent winner selection.
type conversationCache struct {
	mu     sync.Mutex
	values map[string]string
}

func (c *conversationCache) Get(_ context.Context, key string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.values[key], nil
}
func (c *conversationCache) Set(_ context.Context, key, value string, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key] = value
	return nil
}
func (c *conversationCache) SetNX(_ context.Context, key, value string, _ time.Duration) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.values[key]; exists {
		return false, nil
	}
	c.values[key] = value
	return true, nil
}
func (c *conversationCache) Delete(_ context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.values, key)
	return nil
}
func (c *conversationCache) Exists(ctx context.Context, key string) (bool, error) {
	value, err := c.Get(ctx, key)
	return value != "", err
}
func (*conversationCache) LPush(context.Context, string, ...interface{}) error { return nil }
func (*conversationCache) LRange(context.Context, string, int64, int64) ([]string, error) {
	return nil, nil
}
func (*conversationCache) LTrim(context.Context, string, int64, int64) error { return nil }
func (*conversationCache) Close() error                                      { return nil }
func (*conversationCache) Healthy(context.Context) bool                      { return true }

// The Planner passes the same conversation context through multiple tool steps
// and presents the resulting sources to its summarizing model call.
type plannerSearchModel struct{}

func (*plannerSearchModel) Name() string                 { return "planner-script" }
func (*plannerSearchModel) Healthy(context.Context) bool { return true }
func (*plannerSearchModel) ChatStream(context.Context, *model.LLMRequest) (<-chan llm.StreamEvent, error) {
	return nil, fmt.Errorf("unused")
}
func (*plannerSearchModel) Chat(_ context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	if strings.Contains(req.Messages[0].Content, "任务规划专家") {
		input := `{"query":"Go documentation","max_results":1}`
		plan, _ := json.Marshal([]PlanStep{{Step: 1, Tool: "web_search", Input: input}, {Step: 2, Tool: "web_search", Input: input, DependsOn: []int{1}}})
		return &model.LLMResponse{Content: string(plan)}, nil
	}
	if !strings.Contains(req.Messages[len(req.Messages)-1].Content, "https://go.dev/doc/") {
		return nil, fmt.Errorf("planner did not consume source")
	}
	return &model.LLMResponse{Content: "Source: https://go.dev/doc/"}, nil
}
func TestPlannerSearchConversation(t *testing.T) {
	capture := &captureSearchTransport{t: t}
	previous := http.DefaultTransport
	http.DefaultTransport = capture
	t.Cleanup(func() { http.DefaultTransport = previous })
	logger := zap.NewNop()
	registry := tool.NewRegistry()
	registry.MustRegister(builtin.NewSearchTool(logger))
	router := llm.NewRouter(map[string]llm.Client{"planner-script": &plannerSearchModel{}}, "planner-script", config.CBConfig{FailureThreshold: 100, SuccessThreshold: 1})
	planner := NewPlannerAgent(router, tool.NewRouter(registry, logger), logger)
	for i := 0; i < 2; i++ {
		result, err := planner.Execute(context.Background(), "Find Go documentation", nil)
		if err != nil {
			t.Fatal(err)
		}
		if result.Answer != "Source: https://go.dev/doc/" || len(result.ToolCalls) != 2 {
			t.Fatalf("bad planner result: %+v", result)
		}
	}
	if len(capture.ids) != 4 || capture.ids[0] != capture.ids[1] || capture.ids[2] != capture.ids[3] || capture.ids[0] == capture.ids[2] {
		t.Fatalf("wrong standalone run scope: %v", capture.ids)
	}
}
