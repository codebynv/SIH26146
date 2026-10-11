import { escapeHtml, formatNumber, formatPercent, truncateId } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initBehaviorML() {
  unsubscribe = store.subscribe(renderBehaviorML);
  renderBehaviorML(store.state);
}

function renderBehaviorML(state) {
  const container = document.getElementById("behavior-ml-content");
  if (!container) return;
  
  const { behavior, loading, errors } = state;
  
  if (loading || !behavior) {
    container.innerHTML = '<div class="loading-skeleton" style="height: 250px;"></div>';
    return;
  }
  
  if (errors.behavior) {
    container.innerHTML = renderError("behavior", errors.behavior);
    return;
  }
  
  if (behavior.detail) {
    container.innerHTML = `<div class="section-empty">${escapeHtml(behavior.detail)}</div>`;
    return;
  }
  
  container.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: var(--spacing-md);">
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Profile</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          <div><span class="mono">${escapeHtml(behavior.profile_id)}</span></div>
          <div><span class="mono">${behavior.feature_count}</span> features across <span class="mono">${behavior.feature_categories?.length ?? 0}</span> categories</div>
          <div>Grouping: <span class="mono">${escapeHtml(behavior.grouping)}</span></div>
        </div>
      </div>
      
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Feature Categories</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--spacing-xs);">
          ${(behavior.feature_categories || []).map(cat => `<span class="evidence-type-tag">${escapeHtml(cat)}</span>`).join("") || '<span style="color: var(--text-muted); font-size: 11px;">—</span>'}
        </div>
      </div>
      
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Behavioral Unusualness</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          ${behavior.anomaly_band ? `<div>Band: <span class="badge badge--priority">${escapeHtml(behavior.anomaly_band)}</span></div>` : ""}
          ${behavior.anomaly_normalized_score !== null && behavior.anomaly_normalized_score !== undefined ? `<div>Normalized Score: <span class="mono metric__value--priority">${formatPercent(behavior.anomaly_normalized_score)}</span></div>` : ""}
          ${behavior.anomaly_is_in_sample !== null && behavior.anomaly_is_in_sample !== undefined ? `<div>In Sample: <span class="mono">${behavior.anomaly_is_in_sample ? "Yes" : "No"}</span></div>` : ""}
          ${!behavior.anomaly_band && behavior.anomaly_normalized_score === undefined ? '<div class="section-empty">No anomaly context available.</div>' : ""}
        </div>
      </div>
      
      <div class="card">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">Cluster Context</div>
        <div style="display: grid; gap: var(--spacing-xs); font-size: 12px;">
          ${behavior.cluster_id ? `<div>Cluster: <span class="mono">${escapeHtml(truncateId(behavior.cluster_id))}</span></div>` : ""}
          ${behavior.cluster_size !== null && behavior.cluster_size !== undefined ? `<div>Size: <span class="mono">${behavior.cluster_size}</span> members</div>` : ""}
          ${behavior.cluster_distance !== null && behavior.cluster_distance !== undefined ? `<div>Distance to Centroid: <span class="mono">${formatNumber(behavior.cluster_distance)}</span></div>` : ""}
          ${behavior.cluster_normalized !== null && behavior.cluster_normalized !== undefined ? `<div>Normalized (context only): <span class="mono metric__value--context">${formatPercent(behavior.cluster_normalized)}</span></div>` : ""}
          ${!behavior.cluster_id ? '<div class="section-empty">No cluster assignment (noise or unavailable).</div>' : ""}
        </div>
      </div>
      
      ${behavior.explanation_items && behavior.explanation_items.length > 0 ? `
      <div class="card" style="grid-column: 1 / -1;">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--spacing-xs);">ML Explanations</div>
        <ul style="margin: 0; padding-left: var(--spacing-md); font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
          ${behavior.explanation_items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
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

export function cleanupBehaviorML() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
