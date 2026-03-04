import { useState, memo } from "react";

interface DebugInfo {
  sql?: string;
  row_count?: number;
}

interface DebugPanelProps {
  debug?: DebugInfo;
}

/**
 * Collapsible panel showing debug information (SQL query, row count).
 * Only renders if debug info is available.
 */
export const DebugPanel = memo(function DebugPanel({ debug }: DebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!debug?.sql) {
    return null;
  }

  return (
    <div className="debug-panel">
      <button
        type="button"
        className="debug-panel__toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        aria-controls="debug-panel-content"
      >
        <span className="debug-panel__icon">{isExpanded ? "▼" : "▶"}</span>
        <span className="debug-panel__label">Debug Info</span>
        {debug.row_count !== undefined && (
          <span className="debug-panel__badge">{debug.row_count} rows</span>
        )}
      </button>

      {isExpanded && (
        <div id="debug-panel-content" className="debug-panel__content">
          <div className="debug-panel__section">
            <div className="debug-panel__section-label">Generated SQL</div>
            <pre className="debug-panel__code">{debug.sql}</pre>
          </div>
        </div>
      )}
    </div>
  );
});
