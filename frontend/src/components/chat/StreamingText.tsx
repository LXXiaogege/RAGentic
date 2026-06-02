"use client";

interface StreamingTextProps {
  text: string;
}

export function StreamingText({ text }: StreamingTextProps) {
  return (
    <span>
      {text}
      <span className="animate-pulse">▊</span>
    </span>
  );
}
