/**
 * EmailContactList - List of contacts with AI-generated reasons for follow-up.
 */
import type { KeyboardEvent } from "react";
import type { EmailContact } from "../types";

interface EmailContactListProps {
  contacts: EmailContact[];
  category: string;
  loading: boolean;
  generating: boolean;
  selectedContactId: string | null;
  onContactClick: (contactId: string) => void;
  onBack: () => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  quotes: "Open Quotes",
  support: "Support Follow-up",
  renewals: "Renewals",
  recent: "Recent Contacts",
  technical: "Technical Issues",
};

export function EmailContactList({
  contacts,
  category,
  loading,
  generating,
  selectedContactId,
  onContactClick,
  onBack,
}: EmailContactListProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, contactId: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onContactClick(contactId);
    }
  };

  const categoryLabel = CATEGORY_LABELS[category] || category;

  return (
    <div className="email-contact-list">
      <div className="email-contact-list__header">
        <button
          type="button"
          className="email-contact-list__back"
          onClick={onBack}
          aria-label="Go back to questions"
        >
          ← Back
        </button>
        <h3 className="email-contact-list__title">{categoryLabel}</h3>
      </div>

      {loading ? (
        <div className="email-contact-list__loading">
          <div className="email-contact-list__spinner" />
          <span>Finding contacts who need follow-up...</span>
        </div>
      ) : contacts.length === 0 ? (
        <div className="email-contact-list__empty">
          <p>No contacts found for this category.</p>
          <button type="button" onClick={onBack} className="btn btn--primary">
            Try another category
          </button>
        </div>
      ) : (
        <div className="email-contact-list__items" role="list">
          {contacts.map((contact) => (
            <button
              key={contact.contactId}
              className={`email-contact-item ${
                generating && selectedContactId === contact.contactId
                  ? "email-contact-item--generating"
                  : ""
              }`}
              onClick={() => onContactClick(contact.contactId)}
              onKeyDown={(e) => handleKeyDown(e, contact.contactId)}
              type="button"
              disabled={generating}
              role="listitem"
            >
              <div className="email-contact-item__main">
                <span className="email-contact-item__name">{contact.name}</span>
                {contact.company && (
                  <span className="email-contact-item__company">
                    {contact.company}
                  </span>
                )}
              </div>
              <div className="email-contact-item__reason">{contact.reason}</div>
              <div className="email-contact-item__meta">
                <span className="email-contact-item__date">
                  {contact.lastContactAgo || contact.lastContact || "Unknown"}
                </span>
                {generating && selectedContactId === contact.contactId ? (
                  <span className="email-contact-item__status">
                    Generating email...
                  </span>
                ) : (
                  <span className="email-contact-item__action">
                    Click to draft email →
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
