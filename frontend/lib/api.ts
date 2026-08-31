/**
 * Thin, typed fetch wrappers over the live VeriTape API. Every type here
 * comes from lib/api-types.ts (generated from the running API's own
 * OpenAPI schema -- `npm run gen:types`), never hand-written or guessed.
 *
 * No mock data anywhere: every function here hits NEXT_PUBLIC_API_URL for
 * real. ApiError carries enough for a real error UI (status + detail),
 * including the "API isn't running" case (status 0).
 */
import type { components } from "./api-types";

export type PortfolioSummary = components["schemas"]["PortfolioSummaryResponse"];
export type LoanListItem = components["schemas"]["LoanListItem"];
export type LoanListResponse = components["schemas"]["LoanListResponse"];
export type LoanDetail = components["schemas"]["LoanDetailResponse"];
export type ShapDriver = components["schemas"]["ShapDriver"];
export type AuditEntry = components["schemas"]["AuditEntry"];
export type ReviewRequest = components["schemas"]["ReviewRequest"];
export type ReviewResponse = components["schemas"]["ReviewResponse"];
export type ScenarioRequest = components["schemas"]["ScenarioRequest"];
export type ScenarioRunResponse = components["schemas"]["ScenarioRunResponse"];
export type ScenarioTopMover = components["schemas"]["ScenarioTopMover"];
export type AuditVerifyResponse = components["schemas"]["AuditVerifyResponse"];

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, `Could not reach the API at ${API_URL}. Is the server running?`);
  }

  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        // FastAPI 422 validation errors: an array of {loc, msg, ...}
        detail = body.detail.map((e: { msg?: string }) => e.msg).filter(Boolean).join("; ") || detail;
      }
    } catch {
      // response body wasn't JSON -- keep statusText
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

export function getPortfolioSummary(): Promise<PortfolioSummary> {
  return request("/portfolio/summary");
}

export interface LoanListParams {
  page?: number;
  page_size?: number;
  risk_tier?: "low" | "medium" | "high";
  region?: string;
  loan_type?: string;
  flagged?: boolean;
}

/** NB: the API's pagination param is `page_size`, not `limit`. */
export function getLoans(params: LoanListParams = {}): Promise<LoanListResponse> {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) qs.set(key, String(value));
  }
  const query = qs.toString();
  return request(`/loans${query ? `?${query}` : ""}`);
}

export function getLoanDetail(loanId: string): Promise<LoanDetail> {
  return request(`/loans/${encodeURIComponent(loanId)}`);
}

export function submitReview(loanId: string, body: ReviewRequest): Promise<ReviewResponse> {
  return request(`/loans/${encodeURIComponent(loanId)}/review`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function runScenario(body: ScenarioRequest): Promise<ScenarioRunResponse> {
  return request("/scenario/run", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function verifyAudit(): Promise<AuditVerifyResponse> {
  return request("/audit/verify");
}

export function getAuditEntries(): Promise<AuditEntry[]> {
  return request("/audit/entries");
}
