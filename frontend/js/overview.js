import { formatNumber, formatPercent, formatTimestamp, truncateId, escapeHtml, formatStatus, formatRole, getRoleClass, getStatusClass } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initOverview() {
  unsubscribe = store.subscribe(renderOverview);
  renderOverview(store.state);
}

function renderOverview(state) {
  const container = document.getElementById("overview-content");
  if (!container) return;
  
  const { overview, loading, errors } = state;
  
  if (loading || !overview) {
    container.innerHTML = '<div class="loading-skeleton" style="height: 200px;"></div>';
    return;
  }
  
  if (errors.overview) {
    container.innerHTML = renderError("overview", errors.overview);
    return;
  }
  
  const priority = overview.investigative_priority;
  const priorityPercent = priority !== null ? formatPercent(priority) : "—";
  const priorityClass = priority !== null 
    ? (priority >= 0.8 ? "metric__value--high" : priority >= 0.6 ? "metric__value--medium" : priority >= 0.4 ? "metric__value--low" : "metric__value--context")
    : "";
  
  container.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr auto; gap: var(--spacing-lg); align-items: start;">
      <div>
        <div style="display: flex; align-items: baseline; gap: var(--spacing-md); margin-bottom: var(--spacing-md); flex-wrap: wrap;">
          <div>
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Profile ID</div>
            <div class="mono" style="font-size: 14px; font-weight: 600;">${escapeHtml(overview.subject_profile_id || "—")}</div>
          </div>
          <div>
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Assessment ID</div>
            <div class="mono" style="font-size: 12px;">${escapeHtml(truncateId(overview.assessment_id))}</div>
          </div>
          <div>
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Status</div>
            <div>${formatStatus(overview.status)}</div>
          </div>
          ${overview.risk_band ? `<div><div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Risk Band</div><div class="badge badge--priority">${escapeHtml(overview.risk_band)}</div></div>` : ""}
        </div>
        
        <div class="metrics-grid">
          <div class="metric">
            <div class="metric__label">Evidence-Supported Investigative Priority</div>
            <div class="metric__value metric__value--priority ${priorityClass}">${priorityPercent}</div>
          </div>
          <div class="metric">
            <div class="metric__label">Observations</div>
            <div class="metric__value mono">${overview.observation_count ?? 0}</div>
          </div>
          <div class="metric">
            <div class="metric__label">Evidence Relationships</div>
            <div class="metric__value mono">${overview.evidence_relationship_count ?? 0}</div>
          </div>
          <div class="metric">
            <div class="metric__label">Investigation Paths</div>
            <div class="metric__value mono">${overview.investigation_path_count ?? 0}</div>
          </div>
        </div>
      </div>
      
      <div style="display: flex; flex-direction: column; gap: var(--spacing-sm); min-width: 200px;">
        <div class="card">
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Config</div>
          <div class="mono" style="font-size: 11px;">${escapeHtml(overview.config_version || "—")}</div>
        </div>
        ${overview.interpretation ? `
        <div class="card" style="font-size: 11px; color: var(--text-secondary); line-height: 1.5;">
          ${escapeHtml(overview.interpretation)}
        </div>
        ` : ""}
      </div>
    </div>
    
    <div style="margin-top: var(--spacing-lg); display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--spacing-md);">
      ${renderContextCard("Anomaly Context", overview.anomaly_context, [
        { key: "band", label: "Band" },
        { key: "normalized_score", label: "Normalized Score", format: formatNumber },
        { key: "is_in_sample", label: "In Sample", format: v => v ? "Yes" : "No" }
      ])}
      ${renderContextCard("Cluster Context", overview.cluster_context, [
        { key: "cluster_id", label: "Cluster ID", format: truncateId },
        { key: "cluster_size", label: "Cluster Size" },
        { key: "distance_to_centroid", label: "Distance to Centroid", format: formatNumber }
      ])}
      ${renderContextCard("Provenance Context", overview.provenance_context, [
        { key: "cited_edge_count", label: "Cited Edges" },
        { key: "incident_edge_count", label: "Incident Edges" },
        { key: "distinct_source_files", label: "Source Files" }
      ])}
      ${renderContextCard("Temporal Context", overview.temporal_context, [
        { key: "path_count", label: "Paths" },
        { key: "total_transitions", label: "Transitions" },
        { key: "comparable_transitions", label: "Comparable" },
        { key: "forward_transitions", label: "Forward" }
      ])}
    </div>
  `;
}

function renderContextCard(title, context, fields) {
  if (!context || Object.keys(context).length === 0) {
    return `<div class="card"><div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">${escapeHtml(title)}</div><div class="section-empty">No ${title.toLowerCase()} available.</div></div>`;
  }
  
  return `
    <div class="card">
      <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">${escapeHtml(title)}</div>
      <div style="display: grid; gap: var(--spacing-xs);">
        ${fields.map(f => {
          const value = context[f.key];
          if (value === undefined || value === null) return "";
          const formatted = f.format ? f.format(value) : escapeHtml(String(value));
          return `<div style="display: flex; justify-content: space-between; font-size: 12px;"><span class="mono" style="color: var(--text-secondary);">${escapeHtml(f.label)}</span><span>${formatted}</span></div>`;
        }).join("")}
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

export function cleanupOverview() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
