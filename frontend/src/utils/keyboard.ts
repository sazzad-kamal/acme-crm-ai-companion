/**
 * Shared keyboard event utilities for accessibility.
 */
import type { KeyboardEvent } from "react";

/**
 * Create a keyboard handler for Enter/Space activation pattern.
 * Common accessibility pattern for interactive elements.
 */
export function createActivationHandler<T extends HTMLElement>(
  action: () => void
): (e: KeyboardEvent<T>) => void {
  return (e: KeyboardEvent<T>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      action();
    }
  };
}
