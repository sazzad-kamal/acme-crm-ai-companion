/**
 * EmailDraft - Polished email preview with proper formatting.
 * Matches the visual style of Ask AI's message blocks.
 */
import type { GeneratedEmail } from "../types";
import { getInitials, getAvatarColor } from "../utils/avatar";

interface EmailDraftProps {
  email: GeneratedEmail;
  onBack: () => void;
  onReset: () => void;
  onRegenerate?: () => void;
}

export function EmailDraft({ email, onBack, onReset, onRegenerate }: EmailDraftProps) {
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
        <div className="email-draft__secondary-actions">
          {onRegenerate && (
            <button
              type="button"
              className="email-draft__action-btn"
              onClick={onRegenerate}
            >
              🔄 Regenerate
            </button>
          )}
          <button
            type="button"
            className="email-draft__action-btn"
            onClick={onReset}
          >
            ✉️ Draft Another
          </button>
        </div>
      </div>

      <p className="email-draft__hint">
        Your email client will open with this draft ready to review and send.
      </p>
    </div>
  );
}
