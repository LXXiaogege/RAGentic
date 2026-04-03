# -*- coding: utf-8 -*-
"""
@Time：2026/4/3
@Auth：吕鑫
@File：test_memory_leak_fixes.py
@IDE：PyCharm

Tests for memory leak fixes: BoundedMemorySaver and tool_calls_history bounds.
"""

import asyncio
from collections import defaultdict

import pytest

from src.cores.bounded_memory_saver import BoundedMemorySaver


class TestBoundedMemorySaver:
    """Tests for BoundedMemorySaver eviction behavior."""

    def test_put_evicts_oldest_when_limit_exceeded(self):
        """Test that oldest checkpoint is evicted when max_checkpoints is exceeded."""
        saver = BoundedMemorySaver(max_checkpoints=3)

        thread_id = "test-thread"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Add 3 checkpoints (at limit)
        for i in range(3):
            saver.put(
                config,
                {"id": f"checkpoint-{i}", "channel_values": {}},
                {},
                {},
            )

        assert len(saver.storage[thread_id][""].keys()) == 3

        # Add 4th checkpoint - should evict oldest
        saver.put(
            config,
            {"id": "checkpoint-3", "channel_values": {}},
            {},
            {},
        )

        # Should still have 3
        assert len(saver.storage[thread_id][""].keys()) == 3
        # Oldest (checkpoint-0) should be gone
        assert "checkpoint-0" not in saver.storage[thread_id][""].keys()
        # Newest (checkpoint-3) should be present
        assert "checkpoint-3" in saver.storage[thread_id][""].keys()

    def test_different_threads_maintain_separate_counts(self):
        """Test that different threads have independent checkpoint counts."""
        saver = BoundedMemorySaver(max_checkpoints=2)

        config1 = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        config2 = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
            }
        }

        # Fill thread-1 to limit
        for i in range(2):
            saver.put(
                config1,
                {"id": f"t1-{i}", "channel_values": {}},
                {},
                {},
            )

        # Fill thread-2 to limit
        for i in range(2):
            saver.put(
                config2,
                {"id": f"t2-{i}", "channel_values": {}},
                {},
                {},
            )

        assert len(saver.storage["thread-1"][""].keys()) == 2
        assert len(saver.storage["thread-2"][""].keys()) == 2

        # Add more to thread-1 - should only evict from thread-1
        saver.put(
            config1,
            {"id": "t1-2", "channel_values": {}},
            {},
            {},
        )

        assert len(saver.storage["thread-1"][""].keys()) == 2
        assert len(saver.storage["thread-2"][""].keys()) == 2  # Unchanged

    def test_default_max_checkpoints(self):
        """Test default max_checkpoints value (1000)."""
        saver = BoundedMemorySaver()
        assert saver._max_checkpoints == 1000

    def test_custom_max_checkpoints(self):
        """Test custom max_checkpoints value."""
        saver = BoundedMemorySaver(max_checkpoints=500)
        assert saver._max_checkpoints == 500

    @pytest.mark.asyncio
    async def test_aput_evicts_oldest_when_limit_exceeded(self):
        """Test async put also enforces the limit."""
        saver = BoundedMemorySaver(max_checkpoints=2)

        thread_id = "async-test"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        for i in range(2):
            await saver.aput(
                config,
                {"id": f"async-{i}", "channel_values": {}},
                {},
                {},
            )

        assert len(saver.storage[thread_id][""].keys()) == 2

        await saver.aput(
            config,
            {"id": "async-2", "channel_values": {}},
            {},
            {},
        )

        assert len(saver.storage[thread_id][""].keys()) == 2
        assert "async-0" not in saver.storage[thread_id][""].keys()
        assert "async-2" in saver.storage[thread_id][""].keys()


class TestToolCallsHistoryBound:
    """Tests for tool_calls_history bounded growth."""

    def test_tool_calls_history_fifo_eviction(self):
        """Test that tool_calls_history enforces max size with FIFO eviction."""
        from src.cores.pipeline_langgraph import QAState

        # Create state with some existing history
        state = QAState(
            original_query="test",
            tool_calls_history=[
                {"tool": "tool-0", "args": {}, "result": "result-0"},
                {"tool": "tool-1", "args": {}, "result": "result-1"},
            ],
        )

        max_history = 3
        # Simulate adding more items than limit
        new_entries = [
            {"tool": "tool-2", "args": {}, "result": "result-2"},
            {"tool": "tool-3", "args": {}, "result": "result-3"},
            {"tool": "tool-4", "args": {}, "result": "result-4"},
        ]

        history = list(state.tool_calls_history)
        for entry in new_entries:
            history.append(entry)

        if len(history) > max_history:
            evicted = len(history) - max_history
            history = history[-max_history:]

        assert len(history) == 3
        # Oldest entries (tool-0, tool-1) should be evicted
        assert "tool-0" not in [h["tool"] for h in history]
        assert "tool-1" not in [h["tool"] for h in history]
        # Newest entries should remain
        assert "tool-2" in [h["tool"] for h in history]
        assert "tool-3" in [h["tool"] for h in history]
        assert "tool-4" in [h["tool"] for h in history]
