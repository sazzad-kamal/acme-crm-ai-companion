/**
 * EmailSuggestions - Main component for email follow-up workflow.
 *
 * Two-step flow:
 * 1. Click question → Show contact list with AI-generated reasons
 * 2. Click contact → Show draft email with mailto: link
 */
import { useEffect } from "react";
import { useEmailSuggestions } from "../hooks/useEmailSuggestions";
import { EmailQuestions } from "./EmailQuestions";
import { EmailContactList } from "./EmailContactList";
import { EmailDraft } from "./EmailDraft";

export function EmailSuggestions() {
  const {
    view,
    questions,
    selectedCategory,
    contacts,
    generatedEmail,
    loading,
    generating,
    generatingContactId,
    error,
    fetchQuestions,
    fetchContacts,
    generateEmail,
    goBack,
    reset,
  } = useEmailSuggestions();

  // Fetch questions on mount
  useEffect(() => {
    fetchQuestions();
  }, [fetchQuestions]);

  const handleQuestionClick = (categoryId: string) => {
    fetchContacts(categoryId);
  };

  const handleContactClick = (contactId: string) => {
    if (selectedCategory) {
      generateEmail(contactId, selectedCategory);
    }
  };

  return (
    <div className="email-suggestions">
      {error && (
        <div className="email-suggestions__error" role="alert">
          <span className="email-suggestions__error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {view === "questions" && (
        <EmailQuestions
          questions={questions}
          loading={loading}
          onQuestionClick={handleQuestionClick}
        />
      )}

      {view === "contacts" && selectedCategory && (
        <EmailContactList
          contacts={contacts}
          category={selectedCategory}
          loading={loading}
          generating={generating}
          selectedContactId={generatingContactId}
          onContactClick={handleContactClick}
          onBack={goBack}
        />
      )}

      {view === "draft" && generatedEmail && (
        <EmailDraft email={generatedEmail} onBack={goBack} onReset={reset} />
      )}
    </div>
  );
}
