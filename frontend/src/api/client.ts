export type Meta = {
  categories: string[];
  investment_categories: string[];
  non_investment_categories: string[];
  risk_liquidity_profile: Record<
    string,
    { risk: number; liquidity: number; expected_return: number }
  >;
  volatility_assumptions: Record<string, number>;
};

export type DashboardRequest = {
  interest_rates?: Record<string, number> | null;
  months_to_analyze?: number;
  projection_years?: number;
  mc_simulations?: number;
  scenario?: {
    enabled?: boolean;
    savings_adjustment_pct?: number;
    extra_monthly?: number;
    interest_shift_pp?: number;
  };
  goal?: {
    target_amount?: number;
    years?: number;
  };
};

export type DashboardResponse = {
  has_data: boolean;
  message?: string;
  latest?: {
    as_of: string;
    balances: Record<string, number>;
    total: number;
  };
  timeseries?: {
    daily: Array<Record<string, string | number>>;
  };
  monthly?: {
    values: Array<Record<string, string | number>>;
    changes_long: Array<{ month: string; category: string; change: number }>;
    avg_savings: Record<string, number>;
    streak: { current: number; best: number };
  };
  kpis?: {
    liquidity_total: number;
    hhi: number;
    effective_positions: number;
    weighted_risk: number;
    weighted_liquidity: number;
    weighted_expected_return: number;
    max_drawdown: number;
    cagr_total: number;
    vol_monthly: number;
    annual_passive_income: number;
  };
  allocation?: Record<string, number>;
  risk_liquidity?: Array<{
    category: string;
    balance: number;
    risk: number;
    liquidity: number;
    expected_return: number;
  }>;
  correlation?: { categories: string[]; matrix: number[][] };
  projection?: {
    years: number;
    baseline: Array<{ date: string; value: number }>;
    scenario: Array<{ date: string; value: number }> | null;
    monte_carlo: {
      baseline: Array<{ date: string; p05: number; p50: number; p95: number }>;
      scenario:
        | Array<{ date: string; p05: number; p50: number; p95: number }>
        | null;
    };
  };
  goal?: {
    target_amount: number;
    years: number;
    projection_total: number;
    extra_monthly_needed: number;
    on_track: boolean;
  };
  insights?: Array<{ title: string; text: string }>;
};

export type HistoryEntryCreate = {
  category: string;
  amount: number;
  date?: string;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (res.ok) return (await res.json()) as T;

  let message = `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as any;
    message = body?.detail ?? body?.message ?? message;
  } catch {
    // ignore
  }
  throw new Error(message);
}

export async function fetchMeta(): Promise<Meta> {
  return apiFetch<Meta>("/api/v1/meta");
}

export async function fetchDashboard(
  req: DashboardRequest,
): Promise<DashboardResponse> {
  return apiFetch<DashboardResponse>("/api/v1/analytics/dashboard", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function addHistoryEntry(
  entry: HistoryEntryCreate,
): Promise<void> {
  await apiFetch("/api/v1/history", {
    method: "POST",
    body: JSON.stringify(entry),
  });
}

