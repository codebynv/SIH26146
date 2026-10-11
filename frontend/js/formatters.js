const INFERENCE_LABELS = {
  OBSERVED: "OBSERVED",
  CORRELATED: "CORRELATED", 
  INFERRED: "INFERRED",
  REQUIRES_VERIFICATION: "REQUIRES_VERIFICATION"
};

const INFERENCE_CLASSES = {
  OBSERVED: "badge--observed",
  CORRELATED: "badge--correlated",
  INFERRED: "badge--inferred",
  REQUIRES_VERIFICATION: "badge--requires"
};

const ROLE_LABELS = {
  PRIORITY_SIGNAL: "PRIORITY_SIGNAL",
  EVIDENCE_STRENGTH: "EVIDENCE_STRENGTH",
  CONTEXT_ONLY: "CONTEXT_ONLY"
};

const ROLE_CLASSES = {
  PRIORITY_SIGNAL: "badge--priority",
  EVIDENCE_STRENGTH: "badge--evidence",
  CONTEXT_ONLY: "badge--context"
};

const STATUS_LABELS = {
  PRESENT: "PRESENT",
  ABSENT: "ABSENT",
  UNAVAILABLE: "UNAVAILABLE",
  NOT_APPLICABLE: "NOT_APPLICABLE"
};

const STATUS_CLASSES = {
  PRESENT: "badge--present",
  ABSENT: "badge--absent",
  UNAVAILABLE: "badge--unavailable",
  NOT_APPLICABLE: "badge--na"
};

const SIGNAL_FAMILY_ORDER = [
  "ANOMALY",
  "BEHAVIORAL", 
  "CLUSTER_CONTEXT",
  "EVIDENCE_GRAPH",
  "PATH_STRUCTURE",
  "PROVENANCE",
  "TEMPORAL"
];

const SIGNAL_FAMILY_LABELS = {
  ANOMALY: "ANOMALY",
  BEHAVIORAL: "BEHAVIORAL",
  CLUSTER_CONTEXT: "CLUSTER_CONTEXT",
  EVIDENCE_GRAPH: "EVIDENCE_GRAPH",
  PATH_STRUCTURE: "PATH_STRUCTURE",
  PROVENANCE: "PROVENANCE",
  TEMPORAL: "TEMPORAL"
};

export function formatNumber(value, decimals = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(decimals);
}

export function formatPercent(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return (Number(value) * 100).toFixed(decimals) + "%";
}

export function formatTimestamp(isoString) {
  if (!isoString) return "—";
  try {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toISOString().replace("T", " ").replace("Z", " UTC");
  } catch {
    return "—";
  }
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const secs = Number(seconds);
  if (secs < 60) return `${secs.toFixed(1)}s`;
  if (secs < 3600) return `${(secs / 60).toFixed(1)}m`;
  if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`;
  return `${(secs / 86400).toFixed(1)}d`;
}

export function formatInferenceLevel(level) {
  const label = INFERENCE_LABELS[level] || level;
  const cls = INFERENCE_CLASSES[level] || "badge--absent";
  return `<span class="badge inference-badge ${cls}">${label}</span>`;
}

export function formatInferenceLevelText(level) {
  return INFERENCE_LABELS[level] || level;
}

export function getInferenceClass(level) {
  return INFERENCE_CLASSES[level] || "badge--absent";
}

export function formatRole(role) {
  const label = ROLE_LABELS[role] || role;
  const cls = ROLE_CLASSES[role] || "badge--absent";
  return `<span class="badge signal-role ${cls}">${label}</span>`;
}

export function formatRoleText(role) {
  return ROLE_LABELS[role] || role;
}

export function getRoleClass(role) {
  return ROLE_CLASSES[role] || "badge--absent";
}

export function formatStatus(status) {
  const label = STATUS_LABELS[status] || status;
  const cls = STATUS_CLASSES[status] || "badge--absent";
  return `<span class="badge ${cls}">${label}</span>`;
}

export function formatStatusText(status) {
  return STATUS_LABELS[status] || status;
}

export function getStatusClass(status) {
  return STATUS_CLASSES[status] || "badge--absent";
}

export function formatFamily(family) {
  return SIGNAL_FAMILY_LABELS[family] || family;
}

export function sortSignalsByFamily(signals) {
  const familyOrder = { ...Object.fromEntries(SIGNAL_FAMILY_ORDER.map((f, i) => [f, i])) };
  return [...signals].sort((a, b) => {
    const fa = familyOrder[a.family] ?? 999;
    const fb = familyOrder[b.family] ?? 999;
    if (fa !== fb) return fa - fb;
    return a.signal_id.localeCompare(b.signal_id);
  });
}

export function sortEvidenceByInference(evidence) {
  const inferenceOrder = { OBSERVED: 0, CORRELATED: 1, INFERRED: 2, REQUIRES_VERIFICATION: 3 };
  return [...evidence].sort((a, b) => {
    const ia = inferenceOrder[a.inference_level] ?? 999;
    const ib = inferenceOrder[b.inference_level] ?? 999;
    if (ia !== ib) return ia - ib;
    return a.relationship_id.localeCompare(b.relationship_id);
  });
}

export function sortPathsByHopCount(paths) {
  return [...paths].sort((a, b) => a.hop_count - b.hop_count);
}

export function truncateId(id, length = 16) {
  if (!id) return "";
  if (id.length <= length) return id;
  return id.slice(0, length) + "…";
}

const HTML_ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
};

export function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text).replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}

export function escapeRegExp(text) {
  if (text === null || text === undefined) return "";
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function createElement(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

export { INFERENCE_LABELS, INFERENCE_CLASSES, ROLE_LABELS, ROLE_CLASSES, STATUS_LABELS, STATUS_CLASSES, SIGNAL_FAMILY_ORDER, SIGNAL_FAMILY_LABELS };
