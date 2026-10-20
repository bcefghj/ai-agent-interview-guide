// Package memory 提供 AI Agent 的记忆管理能力。
// 采用分层架构：短期记忆（Redis）用于上下文窗口，长期记忆（向量数据库）用于知识沉淀。
package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/enterprise/ai-agent-go/internal/model"
)

// Memory 记忆存储接口
type Memory interface {
	// Save 保存一条记忆
	Save(ctx context.Context, entry *model.MemoryEntry) error
	// Load 加载指定会话的记忆列表
	Load(ctx context.Context, sessionID string, limit int) ([]*model.MemoryEntry, error)
	// Search 语义搜索相关记忆
	Search(ctx context.Context, query string, topK int) ([]*model.MemoryEntry, error)
	// Clear 清除指定会话的所有记忆
	Clear(ctx context.Context, sessionID string) error
}

// Manager 记忆管理器，统一管理短期和长期记忆。
// 写入时同时写入短期记忆，并异步沉淀到长期记忆。
// 读取时先从短期记忆获取近期上下文，再从长期记忆补充相关知识。
type Manager struct {
	shortTerm Memory
	longTerm  Memory
}

// NewManager 创建记忆管理器
func NewManager(shortTerm, longTerm Memory) *Manager {
	return &Manager{
		shortTerm: shortTerm,
		longTerm:  longTerm,
	}
}

// SaveMessage 保存用户或助手的消息到记忆系统。
// 短期记忆同步写入，长期记忆异步写入（不阻塞主流程）。
func (m *Manager) SaveMessage(ctx context.Context, sessionID, role, content string) error {
	entry := &model.MemoryEntry{
		SessionID: sessionID,
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	}

	// 短期记忆：同步写入
	if err := m.shortTerm.Save(ctx, entry); err != nil {
		return err
	}

	// 长期记忆：异步写入，不影响主流程
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = m.longTerm.Save(bgCtx, entry)
	}()

	return nil
}

// LoadContext 加载指定会话的对话上下文。
// 返回短期记忆中的最近 N 条记录，转换为 LLM 消息格式。
func (m *Manager) LoadContext(ctx context.Context, sessionID string, limit int) ([]model.LLMMessage, error) {
	entries, err := m.shortTerm.Load(ctx, sessionID, limit)
	if err != nil {
		return nil, err
	}

	messages := make([]model.LLMMessage, 0, len(entries))
	for _, entry := range entries {
		messages = append(messages, model.LLMMessage{
			Role:    entry.Role,
			Content: entry.Content,
		})
	}

	return messages, nil
}

// SearchRelevant 从长期记忆中搜索与查询语义相关的内容。
// 适用于检索历史对话中的相关知识片段，作为 RAG 的补充。
func (m *Manager) SearchRelevant(ctx context.Context, query string, topK int) ([]*model.MemoryEntry, error) {
	return m.longTerm.Search(ctx, query, topK)
}

// ClearSession 清除指定会话的所有记忆（短期 + 长期）
func (m *Manager) ClearSession(ctx context.Context, sessionID string) error {
	if err := m.shortTerm.Clear(ctx, sessionID); err != nil {
		return err
	}
	return m.longTerm.Clear(ctx, sessionID)
}

// SearchSessionID delegates identity persistence to the conversation store. IDs
// survive process restarts and are cleared with the short-term conversation.
func (m *Manager) SearchSessionID(ctx context.Context, sessionID string) (string, error) {
	store, ok := m.shortTerm.(interface {
		SearchSessionID(context.Context, string) (string, error)
	})
	if !ok {
		return "", fmt.Errorf("short-term memory does not support search session persistence")
	}
	return store.SearchSessionID(ctx, sessionID)
}
