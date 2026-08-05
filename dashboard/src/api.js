import { sampleDashboard } from "./sampleData.js";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function analyzeText(text) {
  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Dashboard API request failed");
  }
  const payload = await response.json();
  return payload.dashboard;
}

export function fallbackDashboard() {
  return sampleDashboard;
}

