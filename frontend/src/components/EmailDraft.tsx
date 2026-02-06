/**
 * EmailDraft - Polished email preview with proper formatting.
 * Matches the visual style of Ask AI's message blocks.
 */
import type { GeneratedEmail } from "../types";

interface EmailDraftProps {
  email: GeneratedEmail;
  onBack: () => void;
  onReset: () => void;
}

/** Generate initials from a name */
function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].charAt(0).toUpperCase();
  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

/** Generate a consistent color from a string */
function getAvatarColor(name: string): string {
  const colors = [
    "#6366F1", "#8B5CF6", "#EC4899", "#F59E0B",
    "#10B981", "#3B82F6", "#EF4444", "#14B8A6",
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

export function EmailDraft({ email, onBack, onReset }: EmailDraftProps) {
  // Convert line breaks to paragraphs for better formatting
  const bodyParagraphs = email.body.split(/\n\n+/).filter(Boolean);

  return (
    <div className="email-draft">
      {/* Header with back button and success badge */}
      <div className="email-draft__header">
        <button
          type="button"
          className="email-draft__back"
          onClick={onBack}
          aria-label="Go back to contact list"
        >
          ← Back
        </button>
        <div className="email-draft__badge">
          <span className="email-draft__badge-icon">✨</span>
          <span className="email-draft__badge-text">Draft Ready</span>
        </div>
      </div>

      {/* Email preview card */}
      <div className="email-draft__card">
        {/* Recipient section */}
        <div className="email-draft__recipient">
          <div
            className="email-draft__avatar"
            style={{ backgroundColor: getAvatarColor(email.contact.name) }}
          >
            {getInitials(email.contact.name)}
          </div>
          <div className="email-draft__recipient-info">
            <div className="email-draft__recipient-name">{email.contact.name}</div>
            <div className="email-draft__recipient-email">{email.contact.email}</div>
            {email.contact.company && (
              <div className="email-draft__recipient-company">{email.contact.company}</div>
            )}
          </div>
        </div>

        {/* Subject line */}
        <div className="email-draft__subject">
          <span className="email-draft__subject-label">Subject</span>
          <span className="email-draft__subject-text">{email.subject}</span>
        </div>

        {/* Email body with proper paragraph formatting */}
        <div className="email-draft__body">
          {bodyParagraphs.map((paragraph, index) => (
            <p key={index} className="email-draft__paragraph">
              {paragraph.split('\n').map((line, lineIndex) => (
                <span key={lineIndex}>
                  {line}
                  {lineIndex < paragraph.split('\n').length - 1 && <br />}
                </span>
              ))}
            </p>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="email-draft__actions">
        <a
          href={email.mailtoLink}
          className="email-draft__send-btn"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className="email-draft__send-icon">📧</span>
          <span className="email-draft__send-text">Open in Email Client</span>
          <span className="email-draft__send-arrow">→</span>
        </a>
        <button
          type="button"
          className="email-draft__reset-btn"
          onClick={onReset}
        >
          ↻ Start Over
        </button>
      </div>

      <p className="email-draft__hint">
        Your email client will open with this draft ready to review and send.
      </p>
    </div>
  );
}
