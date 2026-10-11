import { sortSignalsByFamily, formatNumber, formatPercent, escapeHtml, formatRole, formatStatus, getRoleClass, getStatusClass, SIGNAL_FAMILY_ORDER, SIGNAL_FAMILY_LABELS } from "./formatters.js";
import { store } from "./state.js";

let unsubscribe = null;

export function initWhyThisLead() {
  unsubscribe = store.subscribe(renderWhyThisLead);
  renderWhyThisLead(store.state);
}

function renderWhyThisLead(state) {
  const container = document.getElementById("why-lead-content");
  if (!container) return;
  
  const { signals, loading, errors } = state;
  
  if (loading || signals.length === 0) {
    container.innerHTML = '<div class="loading-skeleton" style="height: 300px;"></div>';
    return;
  }
  
  if (errors.signals) {
    container.innerHTML = renderError("signals", errors.signals);
    return;
  }
  
  const sortedSignals = sortSignalsByFamily(signals);
  const signalsByFamily = {};
  for (const signal of sortedSignals) {
    if (!signalsByFamily[signal.family]) signalsByFamily[signal.family] = [];
    signalsByFamily[signal.family].push(signal);
  }
  
  container.innerHTML = SIGNAL_FAMILY_ORDER.map(family => {
    const familySignals = signalsByFamily[family] || [];
    if (familySignals.length === 0) {
      return renderFamilyAccordion(family, []);
    }
    return renderFamilyAccordion(family, familySignals);
  }).join("");
}

function renderFamilyAccordion(family, signals) {
  const familyLabel = SIGNAL_FAMILY_LABELS[family] || family;
  const hasSignals = signals.length > 0;
  
  return `
    <details class="accordion" open>
      <summary class="accordion__header" style="cursor: pointer; list-style: none;">
        <span>${escapeHtml(familyLabel)}</span>
        <span class="accordion__icon mono">▶</span>
      </summary>
      <div class="accordion__content">
        ${hasSignals ? renderSignalTable(signals) : '<div class="section-empty">No signals in this family.</div>'}
      </div>
    </details>
  `;
}

function renderSignalTable(signals) {
  return `
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th style="width: 30%;">Signal</th>
            <th style="width: 15%; text-align: center;">Role</th>
            <th style="width: 15%; text-align: center;">Status</th>
            <th style="width: 15%; text-align: center;">Value</th>
            <th style="width: 25%;">Explanation</th>
          </tr>
        </thead>
        <tbody>
          ${signals.map(signal => `
            <tr>
              <td class="mono">${escapeHtml(signal.signal_type)}</td>
              <td style="text-align: center;">${formatRole(signal.role)}</td>
              <td style="text-align: center;">${formatStatus(signal.status)}</td>
              <td style="text-align: center;" class="mono">${signal.normalized_value !== null && signal.normalized_value !== undefined ? formatPercent(signal.normalized_value) : "—"}</td>
              <td style="font-size: 11px; color: var(--text-secondary); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(signal.explanation)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
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

export function cleanupWhyThisLead() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}
