import type { ChatMessage } from "../types";
import { MessageBlock } from "./MessageBlock";
import "./DemoLayout.css";

const LOADING_MESSAGES: Record<string, string> = {
  "Brief me on my next call": "Checking your calendar...",
  "What should I focus on today?": "Analyzing your priorities...",
  "Who should I contact next?": "Finding contacts to reach out to...",
  "What's urgent?": "Scanning for urgent items...",
  "Catch me up": "Gathering recent activity...",
};

interface DemoLayoutProps {
  questions: string[];
  database: string;
  activeQuestion: string | null;
  onQuestionClick: (q: string) => void;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
}

export function DemoLayout({
  questions,
  database,
  activeQuestion,
  onQuestionClick,
  messages,
  isLoading,
  error,
}: DemoLayoutProps) {
  return (
    <div className="demo-layout">
      <aside className="demo-sidebar">
        <div className="demo-logo">
          <span className="demo-logo__title">CRM</span>
          <span className="demo-logo__subtitle">Assistant</span>
        </div>
        <div className="demo-database">{database}</div>

        <nav className="demo-questions">
          {questions.map((q) => (
            <button
              key={q}
              className={`demo-question ${activeQuestion === q ? "demo-question--active" : ""}`}
              onClick={() => onQuestionClick(q)}
              disabled={isLoading && activeQuestion !== q}
            >
              <span className="demo-question__indicator">
                {activeQuestion === q ? "\u25CF" : "\u25CB"}
              </span>
              {q}
            </button>
          ))}
        </nav>
      </aside>

      <main className="demo-content">
        {!activeQuestion && !error && (
          <p className="demo-empty">Select a question to get started.</p>
        )}

        {isLoading && (
          <div className="demo-loading">
            <div className="skeleton-answer" />
            <p className="demo-loading__message">
              {LOADING_MESSAGES[activeQuestion || ""] || "Loading..."}
            </p>
          </div>
        )}

        {error && <p className="demo-error">{error}</p>}

        {!isLoading && !error && messages.length > 0 && (
          <MessageBlock message={messages[messages.length - 1]} />
        )}
      </main>
    </div>
  );
}
