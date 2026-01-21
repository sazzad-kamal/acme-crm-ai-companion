interface SuggestedActionsProps {
  actions: string[];
}

/**
 * Displays suggested actions as a highlighted callout
 */
export function SuggestedActions({ actions }: SuggestedActionsProps) {
  if (!actions || actions.length === 0) return null;

  return (
    <div
      className="suggested-actions"
      role="complementary"
      aria-label="Suggested actions"
    >
      <span className="suggested-actions__label">Suggested action:</span>
      {actions.map((action, idx) => (
        <span key={idx} className="suggested-actions__item">
          {action}
        </span>
      ))}
    </div>
  );
}
