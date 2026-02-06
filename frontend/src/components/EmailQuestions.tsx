/**
 * EmailQuestions - Grid of clickable question cards for email follow-up categories.
 */
import type { KeyboardEvent } from "react";
import type { EmailQuestion } from "../types";

interface EmailQuestionsProps {
  questions: EmailQuestion[];
  loading: boolean;
  onQuestionClick: (categoryId: string) => void;
}

const QUESTION_ICONS: Record<string, string> = {
  quotes: "💰",
  support: "🛠️",
  renewals: "🔄",
  recent: "📅",
  technical: "⚙️",
};

export function EmailQuestions({
  questions,
  loading,
  onQuestionClick,
}: EmailQuestionsProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, categoryId: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onQuestionClick(categoryId);
    }
  };

  if (loading) {
    return (
      <div className="email-questions">
        <div className="email-questions__loading">Loading questions...</div>
      </div>
    );
  }

  return (
    <div
      className="email-questions"
      role="group"
      aria-label="Email follow-up categories"
    >
      <h3 className="email-questions__title">Who needs a follow-up email?</h3>
      <p className="email-questions__subtitle">
        Select a category to find contacts who need follow-up
      </p>
      <div className="email-questions__grid">
        {questions.map((question) => (
          <button
            key={question.id}
            className="email-question-card"
            onClick={() => onQuestionClick(question.id)}
            onKeyDown={(e) => handleKeyDown(e, question.id)}
            type="button"
          >
            <span className="email-question-card__icon">
              {QUESTION_ICONS[question.id] || "📧"}
            </span>
            <span className="email-question-card__label">{question.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
