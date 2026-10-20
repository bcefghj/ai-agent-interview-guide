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

const plannerSystemPrompt = `你是一个任务规划专家。请将用户的复杂任务分解为可执行的子步骤。

每个步骤应包含：
1. step: 步骤编号
2. description: 步骤描述
3. tool: 需要使用的工具（如不需要工具则为空）
4. input: 工具输入参数
5. depends_on: 依赖的前置步骤编号列表

请以 JSON 数组格式返回执行计划：
[{"step": 1, "description": "...", "tool": "...", "input": "...", "depends_on": []}]

可用工具：%s`

// Plan 执行计划
type Plan struct {
	Steps []PlanStep `json:"steps"`
}

// PlanStep 计划中的单个步骤
type PlanStep struct {
	Step        int    `json:"step"`
	Description string `json:"description"`
	Tool        string `json:"tool"`
	Input       string `json:"input"`
	DependsOn   []int  `json:"depends_on"`
}

// PlannerAgent 规划型 Agent。
// 先将复杂任务分解为子步骤计划，然后按照依赖关系逐步执行。
type PlannerAgent struct {
	router     *llm.Router
	toolRouter *tool.Router
	logger     *zap.Logger
}

// NewPlannerAgent 创建规划 Agent
func NewPlannerAgent(router *llm.Router, toolRouter *tool.Router, logger *zap.Logger) *PlannerAgent {
	return &PlannerAgent{
		router:     router,
		toolRouter: toolRouter,
		logger:     logger,
	}
}

// Execute 执行复杂任务：生成计划 → 逐步执行 → 汇总结果
func (p *PlannerAgent) Execute(ctx context.Context, task string, history []model.LLMMessage) (*AgentResult, error) {
	ctx = tool.EnsureSearchSession(ctx)

	// 阶段一：生成执行计划
	plan, err := p.generatePlan(ctx, task)
	if err != nil {
		return nil, fmt.Errorf("生成执行计划失败: %w", err)
	}

	p.logger.Info("执行计划已生成", zap.Int("steps", len(plan.Steps)))

	// 阶段二：按步骤执行计划
	result := &AgentResult{}
	stepResults := make(map[int]string) // 步骤编号 -> 执行结果

	for _, step := range plan.Steps {
		// 检查依赖是否满足
		for _, dep := range step.DependsOn {
			if _, ok := stepResults[dep]; !ok {
				return nil, fmt.Errorf("步骤 %d 的依赖 %d 未执行", step.Step, dep)
			}
		}

		p.logger.Info("执行计划步骤",
			zap.Int("step", step.Step),
			zap.String("description", step.Description),
		)

		var output string
		if step.Tool != "" {
			// 需要使用工具
			startTime := time.Now()
			toolResult, err := p.toolRouter.Execute(ctx, step.Tool, step.Input)
			elapsed := time.Since(startTime)

			if err != nil {
				output = fmt.Sprintf("错误: %v", err)
			} else {
				output = toolResult.Output
			}

			result.ToolCalls = append(result.ToolCalls, model.ToolCallInfo{
				ToolName: step.Tool,
				Input:    step.Input,
				Output:   output,
				Duration: elapsed.Milliseconds(),
			})
		} else {
			output = step.Description
		}

		stepResults[step.Step] = output
		result.Steps = append(result.Steps, model.AgentStep{
			StepIndex:  step.Step,
			Type:       "action",
			Content:    step.Description,
			ToolName:   step.Tool,
			ToolInput:  step.Input,
			ToolOutput: output,
			Timestamp:  time.Now(),
		})
	}

	// 阶段三：汇总所有步骤结果，生成最终答案
	answer, err := p.summarize(ctx, task, stepResults)
	if err != nil {
		return nil, err
	}
	result.Answer = answer

	return result, nil
}

// generatePlan 调用 LLM 生成执行计划
func (p *PlannerAgent) generatePlan(ctx context.Context, task string) (*Plan, error) {
	tools := p.toolRouter.ListAvailableTools()
	systemPrompt := fmt.Sprintf(plannerSystemPrompt, strings.Join(tools, ", "))

	req := &model.LLMRequest{
		Messages: []model.LLMMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: task},
		},
		Temperature: 0.2,
	}

	resp, err := p.router.Chat(ctx, req)
	if err != nil {
		return nil, err
	}

	var steps []PlanStep
	if err := json.Unmarshal([]byte(resp.Content), &steps); err != nil {
		return nil, fmt.Errorf("解析执行计划失败: %w", err)
	}

	return &Plan{Steps: steps}, nil
}

// summarize 汇总步骤执行结果，生成最终答案
func (p *PlannerAgent) summarize(ctx context.Context, task string, stepResults map[int]string) (string, error) {
	var resultSummary strings.Builder
	for step, output := range stepResults {
		resultSummary.WriteString(fmt.Sprintf("步骤 %d 结果: %s\n", step, output))
	}

	req := &model.LLMRequest{
		Messages: []model.LLMMessage{
			{
				Role:    "system",
				Content: "请根据各步骤的执行结果，为用户的原始任务生成一个完整、清晰的最终回答。",
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("原始任务: %s\n\n执行结果:\n%s", task, resultSummary.String()),
			},
		},
	}

	resp, err := p.router.Chat(ctx, req)
	if err != nil {
		return "", fmt.Errorf("汇总结果失败: %w", err)
	}

	return resp.Content, nil
}
