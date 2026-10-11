import { escapeHtml, truncateId, formatInferenceLevel, getInferenceClass } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initCrossLayer() {
  unsubscribe = store.subscribe(renderCrossLayer);
  renderCrossLayer(store.state);
}

function renderCrossLayer(state) {
  // The panels render into their own container inside the cross-layer section:
  // the graph and its selection details are siblings in that section, so
  // writing the section's innerHTML here would delete both on every load.
  const container = document.getElementById("cross-layer-panels");
  if (!container) return;
  
  const { evidence, loading, errors } = state;
  
  if (loading || evidence.length === 0) {
    container.innerHTML = '<div class="loading-skeleton" style="height: 350px;"></div>';
    return;
  }
  
  if (errors.evidence) {
    container.innerHTML = renderError("evidence", errors.evidence);
    return;
  }
  
  const networkEdges = evidence.filter(e => e.evidence_type.startsWith("SAME_") && 
    ["SAME_IP", "SAME_PORT", "SAME_ASN", "SAME_GEO"].includes(e.evidence_type));
  const blockchainEdges = evidence.filter(e => e.evidence_type.startsWith("SAME_") && 
    ["SAME_TXID", "SAME_ADDRESS", "SAME_INPUT", "SAME_OUTPUT"].includes(e.evidence_type));
  const otherEdges = evidence.filter(e => !networkEdges.includes(e) && !blockchainEdges.includes(e));
  
  container.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-lg);">
      <div>
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-sm);">Network Observations</div>
        ${renderLayerPanel("Network", networkEdges)}
      </div>
      <div>
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-sm);">Blockchain Observations</div>
        ${renderLayerPanel("Blockchain", blockchainEdges)}
      </div>
    </div>
    
    ${otherEdges.length > 0 ? `
    <div style="margin-top: var(--spacing-lg);">
      <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-sm);">Other Correlations</div>
      ${renderLayerPanel("Other", otherEdges)}
    </div>
    ` : ""}
    
    <div style="margin-top: var(--spacing-lg); padding: var(--spacing-md); background: var(--bg-primary); border: 1px solid var(--border); border-radius: var(--radius-sm); font-size: 11px; color: var(--text-secondary);">
      <strong>Note:</strong> Network ↔ Blockchain relationships are evidence associations derived from shared identifiers (IP, TXID, Address). 
      They do not imply identity attribution (IP → Person, Wallet → Organization) unless explicitly present in source data.
    </div>
  `;
}

function renderLayerPanel(title, edges) {
  if (edges.length === 0) {
    return `<div class="card"><div class="section-empty">No ${title.toLowerCase()} correlations.</div></div>`;
  }
  
  return `
    <div class="card">
      <div style="display: flex; flex-direction: column; gap: var(--spacing-xs);">
        ${edges.map(e => `
          <div style="display: flex; align-items: center; gap: var(--spacing-sm); padding: var(--spacing-xs); background: var(--bg-primary); border: 1px solid var(--border-muted); border-radius: var(--radius-sm);">
            <span class="mono" style="min-width: 120px; font-size: 10px;">${escapeHtml(truncateId(e.source_observation_ids?.join(",") || "—", 20))}</span>
            <span class="evidence-type-tag">${escapeHtml(e.evidence_type)}</span>
            <span style="flex: 1; text-align: center;">${formatInferenceLevel(e.inference_level)}</span>
            <span class="mono" style="min-width: 120px; font-size: 10px; text-align: right;">${escapeHtml(truncateId(e.target_observation_ids?.join(",") || "—", 20))}</span>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderError(section, error) {
  return `
    <div class="section-error">
      <span class="section-error__icon">⚠</span>
      <div>
        <div style="font-weight: 600;">Investigation data could not be loaded.</div>
        <button class="primary" onclick="window.dispatchEvent(new CustomEvent('retry-${section}'))">Retry</button>
      </div>
    </div>
  `;
}

export function cleanupCrossLayer() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
