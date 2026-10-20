// Package builtin 提供 AI Agent 的内置工具实现。
package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/enterprise/ai-agent-go/internal/tool"
)

// SearchTool 网络搜索工具
type SearchTool struct {
	logger   *zap.Logger
	client   *http.Client
	endpoint string
}

// NewSearchTool 创建网络搜索工具
func NewSearchTool(logger *zap.Logger) *SearchTool {
	return &SearchTool{logger: logger, client: &http.Client{Timeout: 25 * time.Second}, endpoint: parallelSearchEndpoint}
}

// Name 返回工具名称
func (t *SearchTool) Name() string { return "web_search" }

// Description 返回工具描述
func (t *SearchTool) Description() string {
	return "搜索互联网获取最新信息，适用于需要实时数据或外部知识的查询"
}

// Parameters 返回参数 JSON Schema
func (t *SearchTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "搜索关键词",
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "最大返回结果数",
				"default":     5,
			},
		},
		"required": []string{"query"},
	}
}

// Execute 执行搜索
func (t *SearchTool) Execute(ctx context.Context, input string) (*tool.ToolResult, error) {
	var params struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return tool.NewErrorResult("参数解析失败: " + err.Error()), nil
	}

	params.Query = strings.TrimSpace(params.Query)
	if params.Query == "" {
		return tool.NewErrorResult("搜索关键词不能为空"), nil
	}

	if params.MaxResults <= 0 {
		params.MaxResults = 5
	}

	if params.MaxResults > 20 {
		return tool.NewErrorResult("max_results 不能超过 20"), nil
	}
	t.logger.Info("执行网络搜索", zap.String("query", params.Query))

	ctx, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()
	result, err := t.search(ctx, params.Query, params.MaxResults)
	if err != nil {
		return nil, fmt.Errorf("Parallel 搜索失败: %w", err)
	}
	return tool.NewSuccessResult(result), nil
}
