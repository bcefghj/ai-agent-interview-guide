package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/enterprise/ai-agent-go/internal/llm"
	"github.com/enterprise/ai-agent-go/internal/model"
	"github.com/enterprise/ai-agent-go/internal/tool"
)

// ReAct Agent 的系统 Prompt 模板
const reactSystemPrompt = `你是一个智能助手，使用 ReAct（Reasoning + Acting）方法来解决问题。

在每一步中，你需要：
1. Thought: 分析当前情况，思考下一步应该做什么
2. Action: 如果需要使用工具，指定工具名称和参数
3. Observation: 观察工具返回的结果
4. 重复上述过程直到得出最终答案

可用工具：
%s

响应格式要求：
- 当需要调用工具时，回复格式：
  Thought: <你的思考>
  Action: {"tool": "<工具名>", "input": <工具参数JSON>}

- 当得出最终答案时，回复格式：
  Thought: <你的思考>
  Final Answer: <最终答案>

重要规则：
- 每次只能调用一个工具
- 最多迭代 %d 次
- 如果无法解决问题，直接给出你能给出的最好答案`

// AgentResult Agent 执行结果
type AgentResult struct {
	Answer    string               `json:"answer"`
	Steps     []model.AgentStep    `json:"steps"`
	ToolCalls []model.ToolCallInfo `json:"tool_calls"`
}

// ReActAgent ReAct 推理-行动循环 Agent。
// 实现经典的 Thought → Action → Observation 迭代推理模式。
type ReActAgent struct {
	router        *llm.Router
	toolRouter    *tool.Router
	maxIterations int
	logger        *zap.Logger
}

// NewReActAgent 创建 ReAct Agent
func NewReActAgent(router *llm.Router, toolRouter *tool.Router, maxIterations int, logger *zap.Logger) *ReActAgent {
	return &ReActAgent{
		router:        router,
		toolRouter:    toolRouter,
		maxIterations: maxIterations,
		logger:        logger,
	}
}

// Run 执行 ReAct 推理循环。
// 核心流程：Thought → Action → Observation → ... → Final Answer
func (a *ReActAgent) Run(ctx context.Context, query string, history []model.LLMMessage) (*AgentResult, error) {
	ctx = tool.EnsureSearchSession(ctx)

	// 构造工具描述
	toolsDesc := a.buildToolsDescription()
	systemPrompt := fmt.Sprintf(reactSystemPrompt, toolsDesc, a.maxIterations)

	// 初始化消息列表
	messages := make([]model.LLMMessage, 0, len(history)+10)
	messages = append(messages, model.LLMMessage{Role: "system", Content: systemPrompt})
	messages = append(messages, history...)
	messages = append(messages, model.LLMMessage{Role: "user", Content: query})

	result := &AgentResult{}

	// ReAct 迭代循环
	for i := 0; i < a.maxIterations; i++ {
		a.logger.Info("ReAct 迭代",
			zap.Int("iteration", i+1),
			zap.Int("max", a.maxIterations),
		)

		// 调用 LLM 获取思考和行动
		req := &model.LLMRequest{
			Messages:    messages,
			Temperature: 0.3,
		}

		resp, err := a.router.Chat(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("ReAct 第 %d 轮 LLM 调用失败: %w", i+1, err)
		}

		content := resp.Content

		// 检查是否已到达最终答案
		if finalAnswer := a.extractFinalAnswer(content); finalAnswer != "" {
			result.Answer = finalAnswer
			result.Steps = append(result.Steps, model.AgentStep{
				StepIndex: i + 1,
				Type:      "thought",
				Content:   content,
				Timestamp: time.Now(),
			})
			a.logger.Info("ReAct 得出最终答案", zap.Int("total_steps", i+1))
			return result, nil
		}

		// 解析 Action
		action := a.extractAction(content)
		if action == nil {
			// 没有 Action 也没有 Final Answer，视为最终答案
			result.Answer = content
			return result, nil
		}

		// 记录思考步骤
		result.Steps = append(result.Steps, model.AgentStep{
			StepIndex: i + 1,
			Type:      "thought",
			Content:   a.extractThought(content),
			Timestamp: time.Now(),
		})

		// 执行工具调用
		startTime := time.Now()
		toolResult, err := a.toolRouter.Execute(ctx, action.Tool, action.Input)
		elapsed := time.Since(startTime)

		var observation string
		if err != nil {
			observation = fmt.Sprintf("工具执行错误: %v", err)
		} else if !toolResult.Success {
			observation = fmt.Sprintf("工具返回错误: %s", toolResult.Error)
		} else {
			observation = toolResult.Output
		}

		// 记录工具调用
		result.ToolCalls = append(result.ToolCalls, model.ToolCallInfo{
			ToolName: action.Tool,
			Input:    action.Input,
			Output:   observation,
			Duration: elapsed.Milliseconds(),
		})
		result.Steps = append(result.Steps, model.AgentStep{
			StepIndex:  i + 1,
			Type:       "action",
			ToolName:   action.Tool,
			ToolInput:  action.Input,
			ToolOutput: observation,
			Timestamp:  time.Now(),
		})

		// 将 LLM 回复和工具观测结果添加到消息历史
		messages = append(messages, model.LLMMessage{
			Role:    "assistant",
			Content: content,
		})
		messages = append(messages, model.LLMMessage{
			Role:    "user",
			Content: fmt.Sprintf("Observation: %s", observation),
		})
	}

	// 达到最大迭代次数，请求 LLM 给出总结
	messages = append(messages, model.LLMMessage{
		Role:    "user",
		Content: "你已达到最大推理步数，请根据已有信息给出最终答案。",
	})

	req := &model.LLMRequest{Messages: messages}
	resp, err := a.router.Chat(ctx, req)
	if err != nil {
		result.Answer = "抱歉，推理过程中出现错误，无法给出完整答案。"
		return result, nil
	}
	result.Answer = resp.Content

	return result, nil
}

// actionPayload Action 的 JSON 结构
type actionPayload struct {
	Tool  string `json:"tool"`
	Input string `json:"input"`
}

// extractAction 从 LLM 响应中提取 Action（工具调用）
func (a *ReActAgent) extractAction(content string) *actionPayload {
	// 查找 "Action:" 后的 JSON
	idx := strings.Index(content, "Action:")
	if idx == -1 {
		return nil
	}

	actionStr := strings.TrimSpace(content[idx+len("Action:"):])

	// 查找第一个 JSON 对象
	start := strings.Index(actionStr, "{")
	if start == -1 {
		return nil
	}

	// 简单的括号匹配找到完整 JSON
	depth := 0
	end := -1
	for i := start; i < len(actionStr); i++ {
		switch actionStr[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i + 1
				break
			}
		}
		if end > 0 {
			break
		}
	}

	if end <= start {
		return nil
	}

	jsonStr := actionStr[start:end]
	var action actionPayload
	if err := json.Unmarshal([]byte(jsonStr), &action); err != nil {
		a.logger.Warn("解析 Action JSON 失败", zap.String("json", jsonStr), zap.Error(err))
		return nil
	}

	// 如果 input 是对象，序列化为字符串
	if action.Input == "" {
		var raw map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &raw); err == nil {
			if inputObj, ok := raw["input"]; ok {
				if inputBytes, err := json.Marshal(inputObj); err == nil {
					action.Input = string(inputBytes)
				}
			}
		}
	}

	return &action
}

// extractFinalAnswer 从 LLM 响应中提取最终答案
func (a *ReActAgent) extractFinalAnswer(content string) string {
	idx := strings.Index(content, "Final Answer:")
	if idx == -1 {
		return ""
	}
	return strings.TrimSpace(content[idx+len("Final Answer:"):])
}

// extractThought 从 LLM 响应中提取思考内容
func (a *ReActAgent) extractThought(content string) string {
	idx := strings.Index(content, "Thought:")
	if idx == -1 {
		return content
	}

	thought := content[idx+len("Thought:"):]
	// 截断到 Action 之前
	if actionIdx := strings.Index(thought, "Action:"); actionIdx > 0 {
		thought = thought[:actionIdx]
	}
	return strings.TrimSpace(thought)
}

// buildToolsDescription 构建工具描述文本，嵌入到系统 Prompt 中
func (a *ReActAgent) buildToolsDescription() string {
	tools := a.toolRouter.ListAvailableTools()
	if len(tools) == 0 {
		return "（无可用工具）"
	}

	var sb strings.Builder
	for _, name := range tools {
		sb.WriteString(fmt.Sprintf("- %s\n", name))
	}
	return sb.String()
}
