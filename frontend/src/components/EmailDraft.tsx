/**
 * EmailDraft - Shows generated email with mailto: link.
 */
import type { GeneratedEmail } from "../types";

interface EmailDraftProps {
  email: GeneratedEmail;
  onBack: () => void;
  onReset: () => void;
}

export function EmailDraft({ email, onBack, onReset }: EmailDraftProps) {
  return (
    <div className="email-draft">
      <div className="email-draft__header">
        <button
          type="button"
          className="email-draft__back"
          onClick={onBack}
          aria-label="Go back to contact list"
        >
          ← Back to contacts
        </button>
      </div>

      <div className="email-draft__recipient">
        <span className="email-draft__recipient-label">To:</span>
        <span className="email-draft__recipient-name">{email.contact.name}</span>
        <span className="email-draft__recipient-email">
          &lt;{email.contact.email}&gt;
        </span>
        {email.contact.company && (
          <span className="email-draft__recipient-company">
            at {email.contact.company}
          </span>
        )}
      </div>

      <div className="email-draft__content">
        <div className="email-draft__subject">
          <span className="email-draft__subject-label">Subject:</span>
          <span className="email-draft__subject-text">{email.subject}</span>
        </div>

        <div className="email-draft__body">
          <pre className="email-draft__body-text">{email.body}</pre>
        </div>
      </div>

      <div className="email-draft__actions">
        <a
          href={email.mailtoLink}
          className="btn btn--primary email-draft__send-btn"
          target="_blank"
          rel="noopener noreferrer"
        >
          📧 Open in Email Client
        </a>
        <button
          type="button"
          className="email-draft__new-btn"
          onClick={onReset}
        >
          Start Over
        </button>
      </div>

      <p className="email-draft__hint">
        Clicking &quot;Open in Email Client&quot; will open your default email app with
        this draft ready to send.
      </p>
    </div>
  );
}
