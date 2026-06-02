"use client";

import { useEffect, useRef } from "react";
import { Message } from "@/lib/types";
import { ChatMessage } from "./ChatMessage";

interface ChatContainerProps {
  messages: Message[];
  isStreaming: boolean;
}

export function ChatContainer({ messages, isStreaming }: ChatContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, isStreaming]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto p-4 space-y-4"
    >
      {messages.length === 0 && (
        <div className="text-center text-muted-foreground py-8">
          <p>开始对话吧！</p>
          <p className="text-sm mt-2">在左侧调整设置，然后输入你的问题</p>
        </div>
      )}
      {messages.map((message, index) => (
        <ChatMessage
          key={message.id || index}
          message={message}
          isStreaming={isStreaming && index === messages.length - 1}
        />
      ))}
    </div>
  );
}
