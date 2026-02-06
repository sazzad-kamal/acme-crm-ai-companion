/**
 * Hook for email suggestion workflow.
 *
 * Two-step flow:
 * 1. Click question → fetchContacts → Show contact list with AI-generated reasons
 * 2. Click contact → generateEmail → Show draft email + mailto: link
 */
import { useState, useCallback } from "react";
import { endpoints } from "../config";
import type { EmailQuestion, EmailContact, GeneratedEmail, EmailContactsResponse } from "../types";

type EmailView = "questions" | "contacts" | "draft";

interface UseEmailSuggestionsReturn {
  // State
  view: EmailView;
  questions: EmailQuestion[];
  selectedCategory: string | null;
  selectedContactId: string | null;
  contacts: EmailContact[];
  generatedEmail: GeneratedEmail | null;
  loading: boolean;
  generating: boolean;
  generatingContactId: string | null;
  error: string | null;
  cachedSecondsAgo: number | null;
  refreshing: boolean;

  // Actions
  fetchQuestions: () => Promise<void>;
  fetchContacts: (category: string) => Promise<void>;
  generateEmail: (contactId: string, category: string) => Promise<void>;
  regenerateEmail: () => Promise<void>;
  refreshCache: () => Promise<void>;
  goBack: () => void;
  reset: () => void;
}

export function useEmailSuggestions(): UseEmailSuggestionsReturn {
  const [view, setView] = useState<EmailView>("questions");
  const [questions, setQuestions] = useState<EmailQuestion[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);
  const [contacts, setContacts] = useState<EmailContact[]>([]);
  const [generatedEmail, setGeneratedEmail] = useState<GeneratedEmail | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generatingContactId, setGeneratingContactId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cachedSecondsAgo, setCachedSecondsAgo] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchQuestions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(endpoints.emailQuestions);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: EmailQuestion[] = await res.json();
      setQuestions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch questions");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchContacts = useCallback(async (category: string) => {
    setLoading(true);
    setError(null);
    setSelectedCategory(category);
    try {
      const res = await fetch(`${endpoints.emailContacts}?category=${encodeURIComponent(category)}`);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: EmailContactsResponse = await res.json();
      setContacts(data.contacts);
      setCachedSecondsAgo(data.cachedSecondsAgo);
      setView("contacts");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch contacts");
    } finally {
      setLoading(false);
    }
  }, []);

  const generateEmail = useCallback(async (contactId: string, category: string) => {
    setGenerating(true);
    setGeneratingContactId(contactId);
    setSelectedContactId(contactId);
    setError(null);
    try {
      const res = await fetch(endpoints.emailGenerate, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contactId, category }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: GeneratedEmail = await res.json();
      setGeneratedEmail(data);
      setView("draft");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate email");
    } finally {
      setGenerating(false);
      setGeneratingContactId(null);
    }
  }, []);

  const regenerateEmail = useCallback(async () => {
    if (!selectedContactId || !selectedCategory) return;
    setGenerating(true);
    setError(null);
    try {
      const res = await fetch(endpoints.emailGenerate, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contactId: selectedContactId, category: selectedCategory }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: GeneratedEmail = await res.json();
      setGeneratedEmail(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to regenerate email");
    } finally {
      setGenerating(false);
    }
  }, [selectedContactId, selectedCategory]);

  const refreshCache = useCallback(async () => {
    if (!selectedCategory) return;
    setRefreshing(true);
    setError(null);
    try {
      // First refresh the cache
      await fetch(endpoints.emailRefresh, { method: "POST" });
      // Then re-fetch contacts for current category
      const res = await fetch(`${endpoints.emailContacts}?category=${encodeURIComponent(selectedCategory)}`);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: EmailContactsResponse = await res.json();
      setContacts(data.contacts);
      setCachedSecondsAgo(data.cachedSecondsAgo);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh data");
    } finally {
      setRefreshing(false);
    }
  }, [selectedCategory]);

  const goBack = useCallback(() => {
    if (view === "draft") {
      setGeneratedEmail(null);
      setView("contacts");
    } else if (view === "contacts") {
      setContacts([]);
      setSelectedCategory(null);
      setView("questions");
    }
    setError(null);
  }, [view]);

  const reset = useCallback(() => {
    setView("questions");
    setSelectedCategory(null);
    setSelectedContactId(null);
    setContacts([]);
    setGeneratedEmail(null);
    setError(null);
  }, []);

  return {
    view,
    questions,
    selectedCategory,
    selectedContactId,
    contacts,
    generatedEmail,
    loading,
    generating,
    generatingContactId,
    error,
    cachedSecondsAgo,
    refreshing,
    fetchQuestions,
    fetchContacts,
    generateEmail,
    regenerateEmail,
    refreshCache,
    goBack,
    reset,
  };
}
