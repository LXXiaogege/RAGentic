"use client";

import { useCallback, useState, useEffect } from "react";
import { useChatStore } from "@/lib/store";
import { ask, askStream } from "@/lib/api";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { ChatInput } from "@/components/chat/ChatInput";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { STARTER_EXAMPLES } from "@/lib/types";

export default function ChatPage() {
  const {
    sessionId,
    setSessionId,
    initSessionId,
    messages,
    addMessage,
    updateLastMessage,
    clearMessages,
    settings,
    updateSettings,
    isStreaming,
    setIsStreaming,
    currentStep,
    setCurrentStep,
  } = useChatStore();

  const [isConnected, setIsConnected] = useState(true);
  const [isClient, setIsClient] = useState(false);

  // Initialize session ID on client side only to avoid hydration mismatch
  useEffect(() => {
    initSessionId();
    setIsClient(true);
  }, [initSessionId]);

  const handleNewChat = useCallback(() => {
    setSessionId(Math.random().toString(36).substring(2) + Date.now().toString(36));
    clearMessages();
    setCurrentStep("");
  }, [setSessionId, clearMessages, setCurrentStep]);

  const handleClearMemory = useCallback(() => {
    setSessionId(Math.random().toString(36).substring(2) + Date.now().toString(36));
    setCurrentStep("");
  }, [setSessionId, setCurrentStep]);

  const handleSend = useCallback(
    async (query: string) => {
      const userMessageId = `${Date.now()}-${Math.random()}`;
      addMessage({
        id: userMessageId,
        role: "user",
        content: query,
      });

      setIsStreaming(true);
      setCurrentStep("");

      try {
        if (settings.use_stream) {
          // Streaming mode
          const assistantMessageId = `${Date.now()}-${Math.random()}-assistant`;
          addMessage({
            id: assistantMessageId,
            role: "assistant",
            content: "",
          });

          let fullContent = "";

          for await (const event of askStream({
            query,
            session_id: sessionId,
            use_memory: settings.use_memory,
            top_k: settings.top_k,
            use_sparse: settings.use_sparse,
            use_reranker: settings.use_reranker,
            enable_think: settings.enable_think,
            allow_model_fallback: settings.allow_model_fallback,
          })) {
            if (event.status === "chunk") {
              fullContent += event.content || "";
              updateLastMessage(fullContent);
            } else if (event.status === "processing") {
              const nodeNames: Record<string, string> = {
                retrieve_knowledge: "检索知识库",
                call_tools: "调用工具",
                transform_query: "改写查询",
                build_context: "构建上下文",
                generate_answer: "生成答案",
              };
              setCurrentStep(nodeNames[event.node || ""] || event.node || "");
            } else if (event.status === "complete") {
              updateLastMessage(event.answer || fullContent, {
                context: event.context || "",
                answer_basis: event.answer_basis,
                sources: event.sources,
                tool_traces: event.tool_traces,
                memory_hits: event.memory_hits,
              });
            } else if (event.status === "error") {
              updateLastMessage(`❌ 错误：${event.error}`);
            }
          }
        } else {
          // Non-streaming mode
          const result = await ask({
            query,
            session_id: sessionId,
            use_memory: settings.use_memory,
            top_k: settings.top_k,
            use_sparse: settings.use_sparse,
            use_reranker: settings.use_reranker,
            enable_think: settings.enable_think,
            allow_model_fallback: settings.allow_model_fallback,
          });

          const assistantMessageId = `${Date.now()}-${Math.random()}-assistant`;
          addMessage({
            id: assistantMessageId,
            role: "assistant",
            content: result.error || result.answer,
            context: result.context || result.kb_context,
            answer_basis: result.answer_basis,
            sources: result.sources,
            tool_traces: result.tool_traces,
            memory_hits: result.memory_hits,
          });
        }
      } catch (error) {
        const assistantMessageId = `${Date.now()}-${Math.random()}-assistant`;
        addMessage({
          id: assistantMessageId,
          role: "assistant",
          content: `❌ 请求失败：${error instanceof Error ? error.message : "未知错误"}`,
        });
        setIsConnected(false);
      } finally {
        setIsStreaming(false);
        setCurrentStep("");
      }
    },
    [
      sessionId,
      settings,
      addMessage,
      updateLastMessage,
      setIsStreaming,
      setCurrentStep,
    ]
  );

  const handleStarterClick = (example: string) => {
    handleSend(example);
  };

  return (
    <div className="flex h-screen">
      {/* Left Sidebar - Settings */}
      <aside className="w-80 border-r bg-muted/20 p-4 flex flex-col gap-4">
        <SettingsPanel
          settings={settings}
          onSettingsChange={updateSettings}
        />

        <Card className="p-4 space-y-2">
          <h3 className="font-medium text-sm">快捷示例</h3>
          <div className="space-y-2">
            {STARTER_EXAMPLES.map((example, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                className="w-full text-left text-xs h-auto py-2 justify-start"
                onClick={() => handleStarterClick(example)}
                disabled={isStreaming}
              >
                {example.length > 30 ? example.slice(0, 30) + "..." : example}
              </Button>
            ))}
          </div>
        </Card>

        <div className="mt-auto space-y-2">
          <Button
            variant="outline"
            className="w-full"
            onClick={handleNewChat}
            disabled={isStreaming}
          >
            🆕 新对话
          </Button>
          <Button
            variant="outline"
            className="w-full"
            onClick={handleClearMemory}
            disabled={isStreaming}
          >
            🗑️ 清空记忆
          </Button>
        </div>

        {!isConnected && (
          <div className="text-xs text-destructive">
            ⚠️ API 未连接，请确保 api_server.py 已启动
          </div>
        )}

        {isClient && sessionId && (
          <div className="text-xs text-muted-foreground">
            会话 ID: {sessionId.slice(0, 8)}...
          </div>
        )}
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        <header className="border-b p-4">
          <h1 className="text-xl font-bold">RAGentic 智能助手</h1>
          {currentStep && (
            <p className="text-sm text-muted-foreground">{currentStep}...</p>
          )}
        </header>

        <ChatContainer messages={messages} isStreaming={isStreaming} />

        <footer className="border-t p-4">
          <ChatInput onSend={handleSend} disabled={isStreaming} />
        </footer>
      </main>
    </div>
  );
}
