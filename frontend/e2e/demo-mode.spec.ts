import { test, expect } from "@playwright/test";

/**
 * E2E Tests for Act! Demo Mode
 * Run with: ACME_DEMO_MODE=true npm run test:e2e -- demo-mode.spec.ts
 */

test.describe("Demo Mode Layout", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("shows two-panel layout with sidebar", async ({ page }) => {
    // Sidebar visible
    await expect(page.locator(".demo-sidebar")).toBeVisible();
    await expect(page.locator(".demo-content")).toBeVisible();

    // Logo and branding
    await expect(page.locator(".demo-logo__title")).toHaveText("CRM");
    await expect(page.locator(".demo-logo__subtitle")).toHaveText("Assistant");
  });

  test("shows database name from env", async ({ page }) => {
    const dbName = page.locator(".demo-database");
    await expect(dbName).toBeVisible();
    await expect(dbName).not.toBeEmpty();
  });

  test("shows all 5 question buttons", async ({ page }) => {
    const questions = page.locator(".demo-question");
    await expect(questions).toHaveCount(5);

    // Verify exact questions
    await expect(page.getByText("Brief me on my next call")).toBeVisible();
    await expect(page.getByText("What should I focus on today?")).toBeVisible();
    await expect(page.getByText("Who should I contact next?")).toBeVisible();
    await expect(page.getByText("What's urgent?")).toBeVisible();
    await expect(page.getByText("Catch me up")).toBeVisible();
  });

  test("shows empty state initially", async ({ page }) => {
    await expect(page.locator(".demo-empty")).toBeVisible();
    await expect(page.getByText("Select a question")).toBeVisible();
  });

  test("no text input field in demo mode", async ({ page }) => {
    // Should NOT have the regular chat input
    await expect(
      page.getByRole("textbox", { name: /ask a question/i })
    ).not.toBeVisible();
  });
});

test.describe("Demo Mode Interactions", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("clicking question shows loading then answer", async ({ page }) => {
    const question = page.getByText("Brief me on my next call");
    await question.click();

    // Should show loading state
    await expect(page.locator(".demo-loading")).toBeVisible();
    await expect(page.getByText("Checking your calendar...")).toBeVisible();

    // Should show answer (wait up to 15s for API + LLM)
    await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });
  });

  test("clicking same question resets to empty", async ({ page }) => {
    const question = page.getByText("Brief me on my next call");

    // Click once - get answer
    await question.click();
    await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });

    // Click again - reset to empty
    await question.click();
    await expect(page.locator(".demo-empty")).toBeVisible();
    await expect(page.locator(".message-block")).not.toBeVisible();
  });

  test("clicking different question replaces answer", async ({ page }) => {
    // Click first question
    await page.getByText("Brief me on my next call").click();
    await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });

    // Click different question
    await page.getByText("What's urgent?").click();

    // Should show new loading message
    await expect(page.getByText("Scanning for urgent items...")).toBeVisible();

    // New answer appears
    await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });
  });

  test("active question shows filled indicator", async ({ page }) => {
    const questionButton = page.locator(".demo-question").filter({ hasText: "Brief me on my next call" });
    await questionButton.click();

    await expect(questionButton).toHaveClass(/demo-question--active/);
  });

  test("other buttons disabled while loading", async ({ page }) => {
    await page.getByText("Brief me on my next call").click();

    // While loading, other buttons should be disabled
    const otherQuestion = page.locator(".demo-question").filter({ hasText: "What's urgent?" });
    await expect(otherQuestion).toBeDisabled();
  });
});

test.describe("Demo Mode Latency", () => {
  test("answer arrives within 15 seconds", async ({ page }) => {
    await page.goto("/");

    const start = Date.now();
    await page.getByText("Brief me on my next call").click();
    await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });
    const elapsed = Date.now() - start;

    console.log(`Latency: ${elapsed}ms`);
    expect(elapsed).toBeLessThan(15000);
  });

  test("all 5 questions respond within timeout", async ({ page }) => {
    await page.goto("/");

    const questions = [
      "Brief me on my next call",
      "What should I focus on today?",
      "Who should I contact next?",
      "What's urgent?",
      "Catch me up",
    ];

    for (const q of questions) {
      const start = Date.now();
      await page.getByText(q).click();
      await expect(page.locator(".message-block")).toBeVisible({ timeout: 15000 });
      const elapsed = Date.now() - start;
      console.log(`"${q}": ${elapsed}ms`);
      expect(elapsed).toBeLessThan(15000);

      // Reset before next
      await page.getByText(q).click();
      await expect(page.locator(".demo-empty")).toBeVisible();
    }
  });
});

test.describe("Demo Mode Error Handling", () => {
  test("shows error state on API failure", async ({ page, context }) => {
    // Block API calls to simulate failure
    await context.route("**/api/chat/**", (route) => route.abort());

    await page.goto("/");
    await page.getByText("Brief me on my next call").click();

    // Should show error
    await expect(page.locator(".demo-error")).toBeVisible({ timeout: 15000 });
  });
});
