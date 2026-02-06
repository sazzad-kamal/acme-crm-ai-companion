/**
 * EmailContactList - List of contacts with avatars and AI-generated reasons.
 * Polished design with loading skeletons and better visual hierarchy.
 */
import type { KeyboardEvent } from "react";
import type { EmailContact } from "../types";
import { getInitials, getAvatarColor } from "../utils/avatar";

interface EmailContactListProps {
  contacts: EmailContact[];
  category: string;
  loading: boolean;
  generating: boolean;
  selectedContactId: string | null;
  cachedSecondsAgo: number | null;
  refreshing: boolean;
  onContactClick: (contactId: string) => void;
  onBack: () => void;
  onRefresh: () => void;
}

function formatCacheAge(seconds: number): string {
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes === 1) return "1 min ago";
  return `${minutes} min ago`;
}

const CATEGORY_CONFIG: Record<string, { title: string; description: string }> = {
  quotes: {
    title: "Open Quotes",
    description: "Contacts with pending quotes that need follow-up",
  },
  support: {
    title: "Support Follow-up",
    description: "Customers who recently had support interactions",
  },
  renewals: {
    title: "Upcoming Renewals",
    description: "Contacts with renewals coming up soon",
  },
  recent: {
    title: "Recent Activity",
    description: "Contacts you've engaged with recently",
  },
  dormant: {
    title: "Re-engage Dormant",
    description: "Contacts who haven't heard from you in a while",
  },
  technical: {
    title: "Technical Issues",
    description: "Contacts with unresolved technical matters",
  },
};

function LoadingSkeleton() {
  return (
    <div className="email-contact-list">
      <div className="email-contact-list__header">
        <div className="skeleton-text" style={{ width: 80, height: 20 }} />
        <div>
          <div className="skeleton-text" style={{ width: 180, height: 24, marginBottom: 8 }} />
          <div className="skeleton-text" style={{ width: 280, height: 16 }} />
        </div>
      </div>
      <div className="email-contact-list__items">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="email-contact-skeleton">
            <div className="skeleton-circle" style={{ width: 48, height: 48 }} />
            <div style={{ flex: 1 }}>
              <div className="skeleton-text" style={{ width: "40%", height: 18, marginBottom: 8 }} />
              <div className="skeleton-text" style={{ width: "70%", height: 14, marginBottom: 6 }} />
              <div className="skeleton-text" style={{ width: "30%", height: 12 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ onBack }: { onBack: () => void }) {
  return (
    <div className="email-contact-list__empty">
      <div className="email-contact-list__empty-illustration">
        <svg viewBox="0 0 200 160" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Empty inbox illustration */}
          <rect x="50" y="45" width="100" height="70" rx="8" fill="#F1F5F9" stroke="#CBD5E1" strokeWidth="2"/>
          <path d="M50 53L100 85L150 53" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round"/>
          <path d="M50 107L75 85" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round"/>
          <path d="M150 107L125 85" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round"/>
          {/* Decorative elements */}
          <circle cx="160" cy="40" r="12" fill="#EEF2FF" stroke="#C7D2FE" strokeWidth="2"/>
          <path d="M160 35V45M155 40H165" stroke="#A5B4FC" strokeWidth="2" strokeLinecap="round"/>
          <circle cx="45" cy="35" r="8" fill="#FEF3C7" stroke="#FCD34D" strokeWidth="2"/>
          {/* Sparkles */}
          <path d="M170 70L171.5 74L175.5 75.5L171.5 77L170 81L168.5 77L164.5 75.5L168.5 74L170 70Z" fill="#94A3B8"/>
          <path d="M35 60L36.5 64L40.5 65.5L36.5 67L35 71L33.5 67L29.5 65.5L33.5 64L35 60Z" fill="#CBD5E1"/>
        </svg>
      </div>
      <h3>No contacts found</h3>
      <p>We couldn't find any contacts in this category right now.</p>
      <button type="button" onClick={onBack} className="btn btn--primary">
        Try another category
      </button>
    </div>
  );
}

export function EmailContactList({
  contacts,
  category,
  loading,
  generating,
  selectedContactId,
  cachedSecondsAgo,
  refreshing,
  onContactClick,
  onBack,
  onRefresh,
}: EmailContactListProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, contactId: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onContactClick(contactId);
    }
  };

  const config = CATEGORY_CONFIG[category] || {
    title: category,
    description: "Contacts who need follow-up",
  };

  if (loading) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="email-contact-list">
      <div className="email-contact-list__header">
        <button
          type="button"
          className="email-contact-list__back"
          onClick={onBack}
          aria-label="Go back to categories"
        >
          ← Back
        </button>
        <div className="email-contact-list__header-text">
          <h2 className="email-contact-list__title">{config.title}</h2>
          <p className="email-contact-list__description">{config.description}</p>
        </div>
        {cachedSecondsAgo !== null && (
          <div className="email-contact-list__cache-info">
            <span className="email-contact-list__cache-age">
              Data from {formatCacheAge(cachedSecondsAgo)}
            </span>
            <button
              type="button"
              className="email-contact-list__refresh"
              onClick={onRefresh}
              disabled={refreshing}
              aria-label="Refresh data"
            >
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        )}
      </div>

      {contacts.length === 0 ? (
        <EmptyState onBack={onBack} />
      ) : (
        <>
          <p className="email-contact-list__hint">
            Click a contact to generate a personalized email draft
          </p>
          <div className="email-contact-list__items" role="list">
            {contacts.map((contact) => {
              const isSelected = generating && selectedContactId === contact.contactId;
              return (
                <button
                  key={contact.contactId}
                  className={`email-contact-card ${isSelected ? "email-contact-card--generating" : ""}`}
                  onClick={() => onContactClick(contact.contactId)}
                  onKeyDown={(e) => handleKeyDown(e, contact.contactId)}
                  type="button"
                  disabled={generating}
                  role="listitem"
                >
                  <div
                    className="email-contact-card__avatar"
                    style={{ backgroundColor: getAvatarColor(contact.name) }}
                  >
                    {getInitials(contact.name)}
                  </div>
                  <div className="email-contact-card__content">
                    <div className="email-contact-card__name">{contact.name}</div>
                    {contact.company && (
                      <div className="email-contact-card__company">{contact.company}</div>
                    )}
                    <div className="email-contact-card__reason">{contact.reason}</div>
                  </div>
                  <div className="email-contact-card__meta">
                    <span className="email-contact-card__time">
                      {contact.lastContactAgo || contact.lastContact || ""}
                    </span>
                    {isSelected ? (
                      <span className="email-contact-card__status">
                        <span className="email-contact-card__spinner" />
                        Writing...
                      </span>
                    ) : (
                      <span className="email-contact-card__cta">Draft email →</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
