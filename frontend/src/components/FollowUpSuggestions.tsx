import type { KeyboardEvent } from "react";

interface FollowUpSuggestionsProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

/**
 * Displays follow-up question suggestions as clickable chips
 */
export function FollowUpSuggestions({
  suggestions,
  onSuggestionClick,
}: FollowUpSuggestionsProps) {
  if (!suggestions || suggestions.length === 0) return null;

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, suggestion: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSuggestionClick(suggestion);
    }
  };

  return (
    <div
      className="follow-up-container"
      role="group"
      aria-label="Suggested follow-up questions"
    >
      {suggestions.map((suggestion, idx) => (
        <button
          key={idx}
          className="follow-up-btn"
          onClick={() => onSuggestionClick(suggestion)}
          onKeyDown={(e) => handleKeyDown(e, suggestion)}
          type="button"
        >
          <span className="follow-up-btn__text">{suggestion}</span>
          <span className="follow-up-btn__arrow" aria-hidden="true">→</span>
        </button>
      ))}
    </div>
  );
}
