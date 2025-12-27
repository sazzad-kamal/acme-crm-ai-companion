import { memo, useMemo } from "react";
import { parseMarkdown } from "../utils/markdown";

interface MarkdownTextProps {
  text: string;
  className?: string;
}

/**
 * Simple markdown renderer for chat messages
 * Supports: **bold**, *italic*, `code`, - lists, numbered lists, ### headers
 */
export const MarkdownText = memo(function MarkdownText({ text, className = "" }: MarkdownTextProps) {
  const rendered = useMemo(() => {
    return parseMarkdown(text);
  }, [text]);

  return (
    <div
      className={`markdown-text ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
});
