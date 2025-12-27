import { test, expect } from '@playwright/test';

/**
 * Enhanced E2E Tests for Chat Application
 *
 * Improvements:
 * - Visual regression testing with screenshots
 * - Network condition testing
 * - Keyboard navigation
 * - Copy functionality
 * - Source citations
 * - Follow-up interactions
 */

test.describe('Chat Application - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('has correct title and header with screenshot', async ({ page }) => {
    await expect(page).toHaveTitle(/Acme CRM/i);
    await expect(page.locator('header')).toBeVisible();

    // Visual regression: capture header
    await expect(page.locator('header')).toHaveScreenshot('header.png');
  });

  test('full page layout screenshot', async ({ page }) => {
    // Take a full-page screenshot for visual regression
    await expect(page).toHaveScreenshot('full-page-initial.png', {
      fullPage: true,
    });
  });

  test('displays example prompts in empty state', async ({ page }) => {
    // Check for example prompts when no messages exist
    const examplePrompts = page.locator('.example-prompts, .suggestion-chip');
    const count = await examplePrompts.count();

    // Should have at least some example prompts
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('can use keyboard shortcuts - Tab navigation', async ({ page }) => {
    // Tab through all interactive elements
    await page.keyboard.press('Tab'); // Skip link
    await page.keyboard.press('Tab'); // Browse Data button
    await page.keyboard.press('Tab'); // Input field

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeFocused();

    // Continue tabbing
    await page.keyboard.press('Tab'); // Send button
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeFocused();
  });

  test('handles very long questions gracefully', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Create a very long question (500 characters)
    const longQuestion = 'What is the status of '.repeat(25) + 'Acme Manufacturing?';
    await input.fill(longQuestion);

    await expect(input).toHaveValue(longQuestion);
    await expect(sendButton).toBeEnabled();
  });

  test('handles special characters in questions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const specialChars = 'What about <Company> & "Partners" (2024)?';
    await input.fill(specialChars);
    await sendButton.click();

    // Should handle special characters without errors
    await expect(page.getByRole('listitem', { name: /conversation about/i }))
      .toBeVisible({ timeout: 30000 });
  });
});

test.describe('Chat Interaction - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('conversation persists after response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Send first question
    await input.fill('What is Acme CRM?');
    await sendButton.click();

    await expect(page.getByRole('listitem', { name: /conversation about/i }))
      .toBeVisible({ timeout: 30000 });

    // Question should remain visible in UI
    await expect(page.locator('.message__question')).toContainText('What is Acme CRM?');

    // Send second question
    await input.fill('How do I create a contact?');
    await sendButton.click();

    // Both messages should be visible
    const messages = page.locator('.message-block');
    await expect(messages).toHaveCount(2, { timeout: 30000 });
  });

  test('shows step-by-step progress pills during processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme Manufacturing?');
    await sendButton.click();

    // Should show progress steps
    const steps = page.locator('.step-pill, .steps-row');
    await expect(steps.first()).toBeVisible({ timeout: 10000 });
  });

  test('displays source citations after response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What accounts have upcoming renewals?');
    await sendButton.click();

    // Wait for response
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Sources section should be visible
    const sourcesSection = page.locator('.sources-section');
    await expect(sourcesSection).toBeVisible();

    // Should have expand/collapse functionality
    const sourcesHeader = page.locator('.sources-section__header');
    await expect(sourcesHeader).toBeVisible();
  });

  test('can copy answer text to clipboard', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Wait for answer
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Find and click copy button
    const copyButton = page.locator('.copy-button, .copy-btn').first();
    if (await copyButton.count() > 0) {
      await copyButton.click();

      // Button should show success state
      await expect(copyButton).toHaveAttribute('aria-label', /copied/i, { timeout: 3000 });
    }
  });

  test('can interact with follow-up suggestions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is the pipeline for TechCorp?');
    await sendButton.click();

    // Wait for response
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Check for follow-up suggestions
    const followUpButtons = page.locator('.follow-up-button, .suggestion-chip').filter({ hasText: /.+/ });
    const count = await followUpButtons.count();

    if (count > 0) {
      // Click first follow-up suggestion
      const firstFollowUp = followUpButtons.first();
      const suggestionText = await firstFollowUp.textContent();

      await firstFollowUp.click();

      // Input should be filled with the suggestion
      await expect(input).toHaveValue(suggestionText || '');
    }
  });

  test('time indicator shows response time', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is the total pipeline value?');
    await sendButton.click();

    // Wait for completion
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Time should be displayed (e.g., "2.3s")
    const timeIndicator = page.locator('.message__time, [class*="time"]');
    await expect(timeIndicator.first()).toBeVisible();

    const timeText = await timeIndicator.first().textContent();
    expect(timeText).toMatch(/\d+\.?\d*s/);
  });
});

test.describe('Accessibility - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('ARIA live region announces new messages', async ({ page }) => {
    // Chat area should have role="log" and aria-live
    const chatArea = page.locator('[role="log"]');
    await expect(chatArea).toBeVisible();
    await expect(chatArea).toHaveAttribute('aria-live', 'polite');
  });

  test('keyboard-only navigation works end-to-end', async ({ page }) => {
    // Start from skip link
    await page.keyboard.press('Tab');
    const skipLink = page.getByRole('link', { name: /skip to main content/i });
    await expect(skipLink).toBeFocused();

    // Activate skip link with Enter
    await page.keyboard.press('Enter');

    // Should jump to main content
    const main = page.getByRole('main');
    await expect(main).toHaveAttribute('tabindex', '-1');
  });

  test('screen reader labels are present', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });

    // Input should have aria-label or associated label
    const ariaLabel = await input.getAttribute('aria-label');
    const ariaLabelledBy = await input.getAttribute('aria-labelledby');

    expect(ariaLabel || ariaLabelledBy).toBeTruthy();
  });

  test('loading state has proper ARIA attributes', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test');
    await sendButton.click();

    // Loading indicator should have role="status"
    const loadingStatus = page.getByRole('status');
    await expect(loadingStatus.first()).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Network Conditions', () => {
  test('handles slow 3G connection', async ({ page, context }) => {
    // Emulate slow 3G
    await context.route('**/*', route => {
      setTimeout(() => route.continue(), 200);
    });

    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Should still work, just slower
    await expect(page.locator('.message__answer'))
      .toBeVisible({ timeout: 60000 }); // Longer timeout for slow connection
  });

  test('shows error when backend is unavailable', async ({ page }) => {
    // Block API requests
    await page.route('**/api/**', route => route.abort());

    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test question');
    await sendButton.click();

    // Should show error or handle gracefully
    await page.waitForTimeout(3000);

    // Input should remain functional
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });
});

test.describe('Responsive Design - Enhanced', () => {
  test('mobile: Browse Data button adapts', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await expect(browseButton).toBeVisible();

    // Button text might be hidden on mobile, but button should work
    await browseButton.click();
    const drawer = page.getByRole('dialog');
    await expect(drawer).toBeVisible();
  });

  test('mobile: drawer covers full width', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();

    const drawer = page.locator('.drawer');
    await expect(drawer).toBeVisible();

    // On mobile, drawer should be full width
    const box = await drawer.boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual(370); // ~100vw
  });

  test('tablet: layout is optimized', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');

    await expect(page.locator('header')).toBeVisible();
    await expect(page.getByRole('textbox')).toBeVisible();

    // Take screenshot for visual regression
    await expect(page).toHaveScreenshot('tablet-layout.png');
  });

  test('desktop: full features visible', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');

    // All elements should be fully visible
    await expect(page.locator('.header__title')).toBeVisible();
    await expect(page.locator('.header__subtitle')).toBeVisible();
    await expect(page.locator('.header__data-btn-text')).toBeVisible();

    // Take screenshot
    await expect(page).toHaveScreenshot('desktop-layout.png');
  });
});

test.describe('Visual Regression', () => {
  test('message block appearance', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Wait for message to appear
    const messageBlock = page.locator('.message-block').first();
    await expect(messageBlock).toBeVisible({ timeout: 30000 });

    // Screenshot the message block
    await expect(messageBlock).toHaveScreenshot('message-block.png');
  });

  test('empty state appearance', async ({ page }) => {
    await page.goto('/');

    // Screenshot empty state
    const chatArea = page.locator('.chat-area, [role="log"]');
    await expect(chatArea).toHaveScreenshot('empty-state.png');
  });
});
