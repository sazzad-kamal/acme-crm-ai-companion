/**
 * EmailQuestions - Landing view with illustration and category cards.
 * Matches the polished style of the Ask AI empty state.
 */
import type { KeyboardEvent } from "react";
import type { EmailQuestion } from "../types";

interface EmailQuestionsProps {
  questions: EmailQuestion[];
  loading: boolean;
  onQuestionClick: (categoryId: string) => void;
}

const QUESTION_CONFIG: Record<string, { icon: string; color: string }> = {
  quotes: { icon: "💰", color: "#F59E0B" },
  support: { icon: "🛠️", color: "#6366F1" },
  renewals: { icon: "🔄", color: "#10B981" },
  recent: { icon: "📅", color: "#8B5CF6" },
  dormant: { icon: "💤", color: "#EC4899" },
  technical: { icon: "⚙️", color: "#3B82F6" },
};

function LoadingSkeleton() {
  return (
    <div className="email-questions">
      <div className="email-questions__illustration">
        <div className="skeleton-circle" style={{ width: 120, height: 120 }} />
      </div>
      <div className="skeleton-text" style={{ width: "60%", height: 28, margin: "0 auto 12px" }} />
      <div className="skeleton-text" style={{ width: "80%", height: 18, margin: "0 auto 32px" }} />
      <div className="email-questions__grid">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="skeleton-card" />
        ))}
      </div>
    </div>
  );
}

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
    return <LoadingSkeleton />;
  }

  return (
    <div
      className="email-questions"
      role="region"
      aria-label="Email follow-up categories"
    >
      {/* Illustration */}
      <div className="email-questions__illustration">
        <svg viewBox="0 0 200 160" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Email envelope */}
          <rect x="40" y="50" width="120" height="80" rx="8" fill="#EEF2FF" stroke="#C7D2FE" strokeWidth="2"/>
          <path d="M40 58L100 95L160 58" stroke="#A5B4FC" strokeWidth="2" strokeLinecap="round"/>

          {/* Floating elements */}
          <circle cx="155" cy="45" r="18" fill="#F0FDF4" stroke="#86EFAC" strokeWidth="2"/>
          <path d="M155 38V52M148 45H162" stroke="#10B981" strokeWidth="2" strokeLinecap="round"/>

          <circle cx="50" cy="35" r="12" fill="#FEF3C7" stroke="#FCD34D" strokeWidth="2"/>
          <path d="M50 30L50 36M50 40L50 40.5" stroke="#F59E0B" strokeWidth="2" strokeLinecap="round"/>

          {/* Sparkles */}
          <path d="M170 80L172 85L177 87L172 89L170 94L168 89L163 87L168 85L170 80Z" fill="#6366F1"/>
          <path d="M35 70L36.5 74L40.5 75.5L36.5 77L35 81L33.5 77L29.5 75.5L33.5 74L35 70Z" fill="#10B981"/>
          <path d="M100 25L101 28L104 29L101 30L100 33L99 30L96 29L99 28L100 25Z" fill="#FBBF24"/>
        </svg>
      </div>

      <h2 className="email-questions__title">
        Who needs to hear from you?
      </h2>

      <p className="email-questions__subtitle">
        We'll find the right contacts and draft a personalized email for you.
      </p>

      <div className="email-questions__grid">
        {questions.map((question) => {
          const config = QUESTION_CONFIG[question.id] || { icon: "📧", color: "#6366F1" };
          return (
            <button
              key={question.id}
              className="email-category-card"
              onClick={() => onQuestionClick(question.id)}
              onKeyDown={(e) => handleKeyDown(e, question.id)}
              type="button"
              style={{ "--accent-color": config.color } as React.CSSProperties}
            >
              <span className="email-category-card__icon">{config.icon}</span>
              <span className="email-category-card__label">{question.label}</span>
              <span className="email-category-card__arrow">→</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
