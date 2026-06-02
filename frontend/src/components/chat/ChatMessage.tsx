"use client";

import { Message } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === "user";
  const basisLabels = {
    kb: "知识库",
    tool: "工具",
    memory: "记忆",
    model: "模型常识",
    mixed: "多来源",
  };
  const hasSources = Boolean(message.sources?.length);
  const hasToolTraces = Boolean(message.tool_traces?.length);
  const hasMemoryHits = Boolean(message.memory_hits?.length);
  const hasContext = Boolean(message.context);
  const showContext = !isUser && (hasSources || hasToolTraces || hasMemoryHits || hasContext);

  return (
    <div
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <Card
        className={cn(
          "max-w-[80%] p-4",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted"
        )}
      >
        {!isUser && message.answer_basis && (
          <div className="mb-2 flex flex-wrap gap-2">
            <span className="rounded border border-border/60 px-2 py-0.5 text-xs text-muted-foreground">
              {basisLabels[message.answer_basis] || "模型常识"}
            </span>
          </div>
        )}
        <div className="whitespace-pre-wrap break-words">
          {message.content}
          {isStreaming && (
            <span className="animate-pulse">▊</span>
          )}
        </div>
        {showContext && (
          <div className="mt-2 pt-2 border-t border-border/50 text-xs text-muted-foreground">
            <details className="cursor-pointer">
              <summary className="font-medium">展开上下文</summary>
              <div className="mt-2 space-y-3">
                {hasSources && (
                  <section className="space-y-2">
                    <div className="font-medium">引用来源</div>
                    {message.sources?.slice(0, 5).map((source, index) => (
                      <div key={`${source.id ?? source.title ?? index}`} className="rounded border border-border/50 p-2">
                        <div className="font-medium text-foreground/80">
                          {source.title || source.source || `知识片段 ${index + 1}`}
                        </div>
                        {typeof source.score === "number" && (
                          <div>相关度：{source.score.toFixed(4)}</div>
                        )}
                        <p className="mt-1 whitespace-pre-wrap break-words">
                          {source.text.length > 280 ? `${source.text.slice(0, 280)}...` : source.text}
                        </p>
                      </div>
                    ))}
                  </section>
                )}

                {hasToolTraces && (
                  <section className="space-y-2">
                    <div className="font-medium">工具返回</div>
                    {message.tool_traces?.slice(0, 5).map((trace, index) => (
                      <div key={`${trace.tool}-${index}`} className="rounded border border-border/50 p-2">
                        <div className="font-medium text-foreground/80">{trace.tool}</div>
                        <p className="mt-1 whitespace-pre-wrap break-words">
                          {(trace.result || "").length > 280
                            ? `${(trace.result || "").slice(0, 280)}...`
                            : trace.result}
                        </p>
                      </div>
                    ))}
                  </section>
                )}

                {hasMemoryHits && (
                  <section className="space-y-2">
                    <div className="font-medium">记忆命中</div>
                    {message.memory_hits?.slice(0, 5).map((hit, index) => (
                      <div key={index} className="rounded border border-border/50 p-2">
                        {typeof hit.score === "number" && (
                          <div>相关度：{hit.score.toFixed(4)}</div>
                        )}
                        <p className="mt-1 whitespace-pre-wrap break-words">
                          {hit.text.length > 280 ? `${hit.text.slice(0, 280)}...` : hit.text}
                        </p>
                      </div>
                    ))}
                  </section>
                )}

                {!hasSources && !hasToolTraces && !hasMemoryHits && message.context && (
                  <pre className="whitespace-pre-wrap break-words text-xs">
                    {message.context.length > 500
                      ? `${message.context.slice(0, 500)}...`
                      : message.context}
                  </pre>
                )}
              </div>
            </details>
          </div>
        )}
      </Card>
    </div>
  );
}
