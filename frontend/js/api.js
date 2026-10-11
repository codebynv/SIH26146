const API_BASE = "/api/dashboard";

class ApiError extends Error {
  constructor(message, status, url) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
  }
}

async function fetchWithRetry(url, options = {}, retries = 1) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Accept": "application/json",
          ...options.headers
        }
      });
      
      if (!response.ok) {
        throw new ApiError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          url
        );
      }
      
      return await response.json();
    } catch (error) {
      lastError = error;
      if (attempt < retries && (error instanceof TypeError || error.status >= 500)) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw error;
    }
  }
  throw lastError;
}

export async function getOverview() {
  return fetchWithRetry(`${API_BASE}/overview`);
}

export async function getSignals() {
  const data = await fetchWithRetry(`${API_BASE}/signals`);
  return data.signals || [];
}

export async function getEvidence() {
  const data = await fetchWithRetry(`${API_BASE}/evidence`);
  return data.evidence || [];
}

export async function getPaths() {
  const data = await fetchWithRetry(`${API_BASE}/paths`);
  return data.paths || [];
}

export async function getBehavior() {
  return fetchWithRetry(`${API_BASE}/behavior`);
}

export async function getProvenance() {
  return fetchWithRetry(`${API_BASE}/provenance`);
}

export async function getAll() {
  return fetchWithRetry(`${API_BASE}/all`);
}

export async function getHealth() {
  return fetchWithRetry(`${API_BASE}/health`);
}
