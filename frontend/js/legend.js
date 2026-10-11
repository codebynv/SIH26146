import { escapeHtml } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initLegend() {
  unsubscribe = store.subscribe(renderLegend);
  renderLegend(store.state);
}

function renderLegend(state) {
  const container = document.getElementById("legend-content");
  if (!container) return;
  
  container.innerHTML = `
    <div class="legend-bar" role="navigation" aria-label="Evidence inference level legend">
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--inference-observed);"></span>
        <span class="legend-item__label">OBSERVED — Directly recorded observation</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--inference-correlated);"></span>
        <span class="legend-item__label">CORRELATED — Shared identifier between observations</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--inference-inferred); background-image: repeating-linear-gradient(90deg, var(--inference-inferred), var(--inference-inferred) 4px, transparent 4px, transparent 8px);"></span>
        <span class="legend-item__label">INFERRED — Derived pattern from correlated evidence</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--inference-requires); background-image: repeating-linear-gradient(90deg, var(--inference-requires), var(--inference-requires) 2px, transparent 2px, transparent 4px);"></span>
        <span class="legend-item__label">REQUIRES_VERIFICATION — Additional evidence needed</span>
      </div>
      <div class="legend-item" style="margin-left: auto; border-left: 1px solid var(--border); padding-left: var(--spacing-md);">
        <span class="legend-item__label" style="font-weight: 600;">Signal Roles:</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--accent-warning);"></span>
        <span class="legend-item__label">PRIORITY_SIGNAL — Contributes to investigative priority</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--accent-primary);"></span>
        <span class="legend-item__label">EVIDENCE_STRENGTH — Measures corroboration</span>
      </div>
      <div class="legend-item">
        <span class="legend-item__swatch" style="background: var(--accent-context);"></span>
        <span class="legend-item__label">CONTEXT_ONLY — Explanatory, no score impact</span>
      </div>
    </div>
  `;
}

export function cleanupLegend() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
