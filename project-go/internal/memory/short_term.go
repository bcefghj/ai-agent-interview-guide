package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/enterprise/ai-agent-go/internal/cache"
	"github.com/enterprise/ai-agent-go/internal/model"
)

const (
	shortTermKeyPrefix     = "memory:short:"
	shortTermTTL           = 2 * time.Hour
	searchSessionKeyPrefix = "memory:search_session:"
)

// ShortTermMemory 基于 Redis 的短期记忆实现。
// 使用 Redis List 存储最近的对话消息，自动滑窗保持固定长度。
type ShortTermMemory struct {
	cache    cache.Cache
	maxItems int // 每个会话保留的最大消息数
}

// NewShortTermMemory 创建短期记忆实例
func NewShortTermMemory(cache cache.Cache, maxItems int) *ShortTermMemory {
	return &ShortTermMemory{
		cache:    cache,
		maxItems: maxItems,
	}
}

// Save 将记忆条目追加到 Redis 列表头部，并裁剪保持固定长度
func (s *ShortTermMemory) Save(ctx context.Context, entry *model.MemoryEntry) error {
	key := s.buildKey(entry.SessionID)

	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("序列化记忆条目失败: %w", err)
	}

	// LPUSH + LTRIM 实现滑窗
	if err := s.cache.LPush(ctx, key, string(data)); err != nil {
		return err
	}

	// 保持列表长度不超过上限
	if err := s.cache.LTrim(ctx, key, 0, int64(s.maxItems-1)); err != nil {
		return err
	}

	// 刷新过期时间
	return s.cache.Set(ctx, key+":ttl", "1", shortTermTTL)
}

// Load 加载指定会话最近的 N 条记忆
func (s *ShortTermMemory) Load(ctx context.Context, sessionID string, limit int) ([]*model.MemoryEntry, error) {
	key := s.buildKey(sessionID)

	if limit <= 0 || limit > s.maxItems {
		limit = s.maxItems
	}

	// LRANGE 获取最近的消息（从新到旧）
	items, err := s.cache.LRange(ctx, key, 0, int64(limit-1))
	if err != nil {
		return nil, err
	}

	// 反转为时间正序（从旧到新），符合对话上下文顺序
	entries := make([]*model.MemoryEntry, 0, len(items))
	for i := len(items) - 1; i >= 0; i-- {
		var entry model.MemoryEntry
		if err := json.Unmarshal([]byte(items[i]), &entry); err != nil {
			continue
		}
		entries = append(entries, &entry)
	}

	return entries, nil
}

// Search 短期记忆不支持语义搜索，返回空结果
func (s *ShortTermMemory) Search(_ context.Context, _ string, _ int) ([]*model.MemoryEntry, error) {
	return nil, nil
}

// Clear 清除指定会话的短期记忆
func (s *ShortTermMemory) Clear(ctx context.Context, sessionID string) error {
	key := s.buildKey(sessionID)
	if err := s.cache.Delete(ctx, key); err != nil {
		return err
	}
	return s.cache.Delete(ctx, searchSessionKeyPrefix+sessionID)
}

func (s *ShortTermMemory) buildKey(sessionID string) string {
	return shortTermKeyPrefix + sessionID
}

// SearchSessionID persists a random ID beside the conversation without sharing
// user-controlled conversation keys with the search service. SET NX makes first
// creation atomic across concurrent turns and multiple application instances.
// Like the conversation history, this key lives until Clear removes it.
func (s *ShortTermMemory) SearchSessionID(ctx context.Context, sessionID string) (string, error) {
	key := searchSessionKeyPrefix + sessionID
	id, err := s.cache.Get(ctx, key)
	if err != nil {
		return "", fmt.Errorf("load search session: %w", err)
	}
	if id == "" {
		candidate := uuid.NewString()
		created, err := s.cache.SetNX(ctx, key, candidate, 0)
		if err != nil {
			return "", fmt.Errorf("create search session: %w", err)
		}
		if created {
			return candidate, nil
		}
		id, err = s.cache.Get(ctx, key)
		if err != nil {
			return "", fmt.Errorf("load concurrent search session: %w", err)
		}
	}
	if _, err := uuid.Parse(id); err != nil {
		return "", fmt.Errorf("stored search session is not a UUID")
	}
	return id, nil
}
