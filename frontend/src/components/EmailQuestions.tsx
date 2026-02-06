/**
 * EmailQuestions - Landing view with illustration and category cards.
 * Premium design with SVG icons and polished animations.
 */
import type { KeyboardEvent, ReactNode } from "react";
import type { EmailQuestion } from "../types";

interface EmailQuestionsProps {
  questions: EmailQuestion[];
  loading: boolean;
  onQuestionClick: (categoryId: string) => void;
}

// SVG Icons for each category
const CategoryIcons: Record<string, ReactNode> = {
  quotes: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="1" x2="12" y2="23" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  ),
  support: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  renewals: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
  ),
  billing: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="1" y="4" width="22" height="16" rx="2" ry="2" />
      <line x1="1" y1="10" x2="23" y2="10" />
    </svg>
  ),
  recent: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  ),
  dormant: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  ),
  technical: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
};

const QUESTION_CONFIG: Record<string, { color: string }> = {
  quotes: { color: "#10B981" },
  support: { color: "#6366F1" },
  renewals: { color: "#3B82F6" },
  billing: { color: "#EC4899" },
  recent: { color: "#8B5CF6" },
  dormant: { color: "#64748B" },
  technical: { color: "#0EA5E9" },
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
          const config = QUESTION_CONFIG[question.id] || { color: "#6366F1" };
          const icon = CategoryIcons[question.id] || CategoryIcons.support;
          return (
            <button
              key={question.id}
              className="email-category-card"
              onClick={() => onQuestionClick(question.id)}
              onKeyDown={(e) => handleKeyDown(e, question.id)}
              type="button"
              style={{ "--accent-color": config.color } as React.CSSProperties}
            >
              <span className="email-category-card__icon" style={{ color: config.color }}>
                {icon}
              </span>
              <span className="email-category-card__label">{question.label}</span>
              <span className="email-category-card__arrow">→</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
