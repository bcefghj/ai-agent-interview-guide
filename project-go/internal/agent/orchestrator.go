// Package agent 提供 AI Agent 的编排和推理能力。
// 包含多种 Agent 策略：ReAct（推理-行动循环）、Planner（规划执行）、Reflection（反思改进）。
package agent

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/enterprise/ai-agent-go/internal/config"
	"github.com/enterprise/ai-agent-go/internal/intent"
	"github.com/enterprise/ai-agent-go/internal/llm"
	"github.com/enterprise/ai-agent-go/internal/memory"
	"github.com/enterprise/ai-agent-go/internal/model"
	"github.com/enterprise/ai-agent-go/internal/rag"
	"github.com/enterprise/ai-agent-go/internal/tool"
)

// OrchestratorDeps Agent 编排器的依赖注入容器
type OrchestratorDeps struct {
	ModelRouter      *llm.Router
	MemoryManager    *memory.Manager
	ToolRouter       *tool.Router
	IntentRecognizer *intent.Recognizer
	Retriever        *rag.Retriever
	Reranker         *rag.Reranker
	Generator        *rag.Generator
	Config           config.AgentConfig
	Logger           *zap.Logger
}

// Orchestrator Agent 编排器。
// 作为整个 Agent 系统的入口，负责：
// 1. 意图识别 → 确定处理策略
// 2. 根据意图选择合适的 Agent（ReAct / Planner / 直接回复）
// 3. 管理上下文记忆
// 4. 编排 RAG 检索和工具调用
type Orchestrator struct {
	deps       OrchestratorDeps
	reactAgent *ReActAgent
	planner    *PlannerAgent
	reflector  *ReflectionAgent
}

// NewOrchestrator 创建 Agent 编排器并初始化所有子 Agent
func NewOrchestrator(deps OrchestratorDeps) *Orchestrator {
	o := &Orchestrator{deps: deps}

	o.reactAgent = NewReActAgent(
		deps.ModelRouter,
		deps.ToolRouter,
		deps.Config.MaxIterations,
		deps.Logger,
	)

	o.planner = NewPlannerAgent(
		deps.ModelRouter,
		deps.ToolRouter,
		deps.Logger,
	)

	o.reflector = NewReflectionAgent(
		deps.ModelRouter,
		deps.Logger,
	)

	return o
}

// ProcessMessage 处理用户消息的完整流程：
// 意图识别 → 加载上下文 → 策略路由 → 执行 Agent → 保存记忆 → 返回结果
func (o *Orchestrator) ProcessMessage(ctx context.Context, req *model.ChatRequest) (*model.ChatResponse, error) {
	startTime := time.Now()

	// 1. 意图识别
	intentResult, err := o.deps.IntentRecognizer.Recognize(ctx, req.Message)
	if err != nil {
		o.deps.Logger.Warn("意图识别失败，使用默认策略", zap.Error(err))
		intentResult = &model.IntentResult{Intent: "chat", Confidence: 0.5}
	}

	if intentResult.Intent == intent.IntentToolUse || intentResult.Intent == intent.IntentComplexTask {
		sessionID := req.SessionID
		ctx = tool.WithSearchSessionResolver(ctx, func(searchCtx context.Context) (string, error) {
			return o.deps.MemoryManager.SearchSessionID(searchCtx, sessionID)
		})
	}

	// 2. 加载历史上下文
	history, err := o.deps.MemoryManager.LoadContext(ctx, req.SessionID, 10)
	if err != nil {
		o.deps.Logger.Warn("加载上下文失败", zap.Error(err))
		history = nil
	}

	// 3. 根据意图路由到对应的处理策略
	var answer string
	var toolCalls []model.ToolCallInfo
	var references []model.Reference

	switch intentResult.Intent {
	case intent.IntentRAGQuery:
		answer, references, err = o.handleRAGQuery(ctx, req.Message, history)
	case intent.IntentToolUse:
		answer, toolCalls, err = o.handleToolUse(ctx, req.Message, history)
	case intent.IntentComplexTask:
		answer, toolCalls, err = o.handleComplexTask(ctx, req.Message, history)
	default:
		answer, err = o.handleChat(ctx, req.Message, history)
	}

	if err != nil {
		return nil, err
	}

	// 4. 反思改进（如果启用）
	if o.deps.Config.EnableReflection && len(answer) > 0 {
		improved, reflectErr := o.reflector.Reflect(ctx, req.Message, answer)
		if reflectErr == nil && improved != "" {
			answer = improved
		}
	}

	// 5. 保存本轮对话到记忆
	_ = o.deps.MemoryManager.SaveMessage(ctx, req.SessionID, "user", req.Message)
	_ = o.deps.MemoryManager.SaveMessage(ctx, req.SessionID, "assistant", answer)

	resp := &model.ChatResponse{
		SessionID:  req.SessionID,
		Content:    answer,
		ToolCalls:  toolCalls,
		References: references,
		CreatedAt:  time.Now(),
	}

	o.deps.Logger.Info("消息处理完成",
		zap.String("intent", intentResult.Intent),
		zap.Duration("elapsed", time.Since(startTime)),
	)

	return resp, nil
}

// handleChat 处理普通对话
func (o *Orchestrator) handleChat(ctx context.Context, message string, history []model.LLMMessage) (string, error) {
	messages := append(history, model.LLMMessage{
		Role:    "user",
		Content: message,
	})

	req := &model.LLMRequest{Messages: messages}
	resp, err := o.deps.ModelRouter.Chat(ctx, req)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// handleRAGQuery 处理知识库查询：检索 → 重排 → 生成
func (o *Orchestrator) handleRAGQuery(ctx context.Context, query string, history []model.LLMMessage) (string, []model.Reference, error) {
	// 检索
	refs, err := o.deps.Retriever.Retrieve(ctx, query, 5)
	if err != nil {
		o.deps.Logger.Warn("RAG 检索失败，降级为直接回答", zap.Error(err))
		answer, chatErr := o.handleChat(ctx, query, history)
		return answer, nil, chatErr
	}

	// 重排序
	refs, _ = o.deps.Reranker.Rerank(ctx, query, refs)

	// 生成答案
	answer, err := o.deps.Generator.Generate(ctx, query, refs)
	if err != nil {
		return "", nil, err
	}

	return answer, refs, nil
}

// handleToolUse 处理工具调用：使用 ReAct Agent 进行推理和工具调用
func (o *Orchestrator) handleToolUse(ctx context.Context, message string, history []model.LLMMessage) (string, []model.ToolCallInfo, error) {
	result, err := o.reactAgent.Run(ctx, message, history)
	if err != nil {
		return "", nil, fmt.Errorf("ReAct Agent 执行失败: %w", err)
	}
	return result.Answer, result.ToolCalls, nil
}

// handleComplexTask 处理复杂任务：先规划再逐步执行
func (o *Orchestrator) handleComplexTask(ctx context.Context, message string, history []model.LLMMessage) (string, []model.ToolCallInfo, error) {
	result, err := o.planner.Execute(ctx, message, history)
	if err != nil {
		// 降级到 ReAct
		o.deps.Logger.Warn("规划执行失败，降级到 ReAct", zap.Error(err))
		return o.handleToolUse(ctx, message, history)
	}
	return result.Answer, result.ToolCalls, nil
}
