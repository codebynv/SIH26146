import { escapeHtml, formatNumber, formatPercent, truncateId } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initProvenance() {
  unsubscribe = store.subscribe(renderProvenance);
  renderProvenance(store.state);
}

function renderProvenance(state) {
  const container = document.getElementById("provenance-content");
  if (!container) return;
  
  const { provenance, loading, errors } = state;
  
  if (loading || !provenance) {
    container.innerHTML = '<div class="loading-skeleton" style="height: 250px;"></div>';
    return;
  }
  
  if (errors.provenance) {
    container.innerHTML = renderError("provenance", errors.provenance);
    return;
  }
  
  if (provenance.detail) {
    container.innerHTML = `<div class="section-empty">${escapeHtml(provenance.detail)}</div>`;
    return;
  }
  
  container.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: var(--spacing-md);">
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Source</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          <div>Ingestion: <span class="mono">${escapeHtml(truncateId(provenance.ingestion_id))}</span></div>
          <div>Files: <span class="mono">${(provenance.source_files || []).length}</span></div>
          <div>Records: <span class="mono">${(provenance.source_records || []).length}</span></div>
        </div>
      </div>
      
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Evidence Quality</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          ${provenance.evidence_citation_ratio !== null && provenance.evidence_citation_ratio !== undefined ? `
            <div>Citation Ratio: <span class="mono metric__value--priority">${formatPercent(provenance.evidence_citation_ratio)}</span></div>
            <div>Cited / Incident: <span class="mono">${provenance.cited_edge_count ?? 0} / ${provenance.incident_edge_count ?? 0}</span></div>
          ` : '<div class="section-empty">No citation data.</div>'}
          ${provenance.distinct_source_files !== null && provenance.distinct_source_files !== undefined ? `<div>Distinct Source Files: <span class="mono">${provenance.distinct_source_files}</span></div>` : ""}
        </div>
      </div>
      
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Temporal Verification</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          ${provenance.temporal_comparability !== null && provenance.temporal_comparability !== undefined ? `<div>Comparability: <span class="mono">${formatPercent(provenance.temporal_comparability)}</span></div>` : ""}
          ${provenance.temporal_forward_flow !== null && provenance.temporal_forward_flow !== undefined ? `<div>Forward Flow: <span class="mono">${formatPercent(provenance.temporal_forward_flow)}</span></div>` : ""}
          ${provenance.temporal_comparability === null && provenance.temporal_forward_flow === null ? '<div class="section-empty">No temporal verification data.</div>' : ""}
        </div>
      </div>
      
      <div class="card" style="grid-column: 1 / -1;">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Missing / Unavailable Signals</div>
        ${provenance.missing_signals && provenance.missing_signals.length > 0 ? `
          <div style="display: flex; flex-wrap: wrap; gap: var(--spacing-xs);">
            ${provenance.missing_signals.map(sig => `<span class="badge badge--unavailable">${escapeHtml(sig)}</span>`).join("")}
          </div>
        ` : '<div class="section-empty">All expected signals available.</div>'}
      </div>
      
      ${provenance.source_files && provenance.source_files.length > 0 ? `
      <div class="card" style="grid-column: 1 / -1;">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Source Files</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--spacing-xs); font-size: 11px;">
          ${provenance.source_files.map(f => `<span class="evidence-type-tag">${escapeHtml(f)}</span>`).join("")}
        </div>
      </div>
      ` : ""}
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

export function cleanupProvenance() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
