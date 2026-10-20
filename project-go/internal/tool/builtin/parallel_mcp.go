package builtin

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/enterprise/ai-agent-go/internal/tool"
)

const (
	parallelSearchEndpoint = "https://search.parallel.ai/mcp"
	parallelUserAgent      = "ai-agent-interview-guide/0.1 (+https://github.com/bcefghj/ai-agent-interview-guide)"
	maxMCPResponse         = 2 * 1024 * 1024
)

// Each search uses its own MCP handshake. The HTTP session is separate from the
// conversation ID in tool arguments, and is never shared between conversations.
type searchMCP struct {
	tool     *SearchTool
	session  string
	protocol string
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (t *SearchTool) search(ctx context.Context, query string, limit int) (string, error) {
	sessionID, err := tool.SearchSession(ctx)
	if err != nil {
		return "", fmt.Errorf("search conversation: %w", err)
	}
	m := &searchMCP{tool: t, protocol: "2025-03-26"}
	raw, err := m.request(ctx, 1, "initialize", map[string]interface{}{
		"protocolVersion": m.protocol, "capabilities": map[string]interface{}{},
		"clientInfo": map[string]string{"name": "ai-agent-interview-guide", "version": "0.1.0"},
	})
	if err != nil {
		return "", err
	}
	defer m.close(ctx)
	var initialized struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if err := json.Unmarshal(raw, &initialized); err != nil {
		return "", fmt.Errorf("invalid initialize response: %w", err)
	}
	if initialized.ProtocolVersion != "2025-03-26" && initialized.ProtocolVersion != "2025-06-18" && initialized.ProtocolVersion != "2025-11-25" {
		return "", fmt.Errorf("unsupported MCP protocol %q", initialized.ProtocolVersion)
	}
	m.protocol = initialized.ProtocolVersion
	if _, err := m.request(ctx, 0, "notifications/initialized", nil); err != nil {
		return "", err
	}
	raw, err = m.request(ctx, 2, "tools/list", map[string]interface{}{})
	if err != nil {
		return "", err
	}
	var discovered struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &discovered); err != nil {
		return "", fmt.Errorf("invalid tool list: %w", err)
	}
	found := false
	for _, candidate := range discovered.Tools {
		if candidate.Name == "web_search" {
			found = true
		}
	}
	if !found {
		return "", fmt.Errorf("MCP server does not offer web_search")
	}
	args := map[string]interface{}{"objective": query, "search_queries": []string{query}, "session_id": sessionID}
	raw, err = m.request(ctx, 3, "tools/call", map[string]interface{}{"name": "web_search", "arguments": args})
	if err != nil {
		return "", err
	}
	var result struct {
		IsError bool `json:"isError"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("invalid search response: %w", err)
	}
	var texts []string
	for _, c := range result.Content {
		if c.Type == "text" {
			texts = append(texts, c.Text)
		}
	}
	text := strings.Join(texts, "\n")
	if result.IsError {
		return "", fmt.Errorf("MCP web_search: %s", text)
	}
	// Preserve source URLs and excerpts; max_results applies locally because the
	// anonymous MCP tool has no per-call max_results argument.
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		return "", fmt.Errorf("invalid search data: %w", err)
	}
	var results []json.RawMessage
	if err := json.Unmarshal(payload["results"], &results); err != nil || results == nil {
		return "", fmt.Errorf("search data is missing results")
	}
	if len(results) > limit {
		results = results[:limit]
	}
	payload["results"], _ = json.Marshal(results)
	output, err := json.Marshal(payload)
	return string(output), err
}

// request implements the Streamable HTTP POST response modes used by Search MCP.
// A bounded reader and context apply to both JSON and SSE, including keepalives.
func (m *searchMCP) request(ctx context.Context, id int, method string, params interface{}) (json.RawMessage, error) {
	payload := map[string]interface{}{"jsonrpc": "2.0", "method": method}
	if id != 0 {
		payload["id"] = id
	}
	if params != nil {
		payload["params"] = params
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.tool.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("User-Agent", parallelUserAgent)
	if method != "initialize" {
		req.Header.Set("MCP-Protocol-Version", m.protocol)
	}
	if m.session != "" {
		req.Header.Set("Mcp-Session-Id", m.session)
	}
	resp, err := m.tool.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("MCP HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if method == "initialize" {
		m.session = resp.Header.Get("Mcp-Session-Id")
	}
	if id == 0 {
		return nil, nil
	}
	media, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	reader := io.LimitReader(resp.Body, maxMCPResponse+1)
	if media == "application/json" {
		data, err := io.ReadAll(reader)
		if err != nil {
			return nil, err
		}
		if len(data) > maxMCPResponse {
			return nil, fmt.Errorf("MCP response too large")
		}
		return decodeRPC(data, id)
	}
	if media != "text/event-stream" {
		return nil, fmt.Errorf("unsupported MCP content type %q", media)
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), maxMCPResponse)
	var data []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if len(data) == 0 {
				continue
			}
			encoded := []byte(strings.Join(data, "\n"))
			data = nil
			var envelope struct {
				ID int `json:"id"`
			}
			if err := json.Unmarshal(encoded, &envelope); err != nil {
				return nil, fmt.Errorf("invalid MCP event: %w", err)
			}
			if envelope.ID == id {
				return decodeRPC(encoded, id)
			}
		} else if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("MCP stream ended without response %d", id)
}

func decodeRPC(data []byte, id int) (json.RawMessage, error) {
	var response rpcResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("invalid MCP JSON: %w", err)
	}
	if response.JSONRPC != "2.0" || response.ID != id {
		return nil, fmt.Errorf("MCP response identity mismatch")
	}
	if response.Error != nil {
		return nil, fmt.Errorf("MCP error %d: %s", response.Error.Code, response.Error.Message)
	}
	if len(response.Result) == 0 || string(response.Result) == "null" {
		return nil, fmt.Errorf("MCP result is missing")
	}
	return response.Result, nil
}

// Release server-side transport state when a server assigns an HTTP session.
func (m *searchMCP) close(parent context.Context) {
	if m.session == "" {
		return
	}
	ctx, cancel := context.WithTimeout(parent, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, m.tool.endpoint, nil)
	if err != nil {
		return
	}
	req.Header.Set("Mcp-Session-Id", m.session)
	req.Header.Set("MCP-Protocol-Version", m.protocol)
	req.Header.Set("User-Agent", parallelUserAgent)
	resp, err := m.tool.client.Do(req)
	if err == nil {
		resp.Body.Close()
	}
}
