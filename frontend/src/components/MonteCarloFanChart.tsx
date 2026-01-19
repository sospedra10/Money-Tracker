import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency } from "../lib/format";

type Row = { date: string; p05: number; p50: number; p95: number };

export function MonteCarloFanChart({
  baseline,
  scenario,
}: {
  baseline: Row[];
  scenario: Row[] | null;
}) {
  if (!baseline.length) return null;

  const data = baseline.map((r, idx) => {
    const scenarioRow = scenario?.[idx];
    return {
      date: r.date,
      p05: r.p05,
      p50: r.p50,
      p95: r.p95,
      band: r.p95 - r.p05,
      scenario_p50: scenarioRow?.p50 ?? null,
    };
  });

  const tooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: any[];
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]?.payload as
      | {
          date: string;
          p05: number;
          p50: number;
          p95: number;
          scenario_p50: number | null;
        }
      | undefined;
    if (!row) return null;

    return (
      <div
        style={{
          background: "rgba(20,20,20,0.9)",
          border: "1px solid rgba(255,255,255,0.12)",
          padding: 10,
          borderRadius: 10,
          minWidth: 220,
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 6 }}>{label ?? row.date}</div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
          <span>p05</span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            {formatCurrency(row.p05)}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
          <span>p50</span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            {formatCurrency(row.p50)}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
          <span>p95</span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            {formatCurrency(row.p95)}
          </span>
        </div>
        {row.scenario_p50 != null ? (
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 12,
              marginTop: 6,
              paddingTop: 6,
              borderTop: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            <span>Escenario p50</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>
              {formatCurrency(row.scenario_p50)}
            </span>
          </div>
        ) : null}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={360}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis
          tick={{ fontSize: 12 }}
          tickFormatter={(v) => formatCurrency(Number(v))}
          width={90}
        />
        <Tooltip content={tooltip} />
        <Legend />
        <Area
          type="monotone"
          dataKey="p05"
          stackId="1"
          stroke="transparent"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="band"
          stackId="1"
          stroke="transparent"
          fill="#4dabf7"
          fillOpacity={0.22}
          name="5%-95% (baseline)"
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="p50"
          stroke="#ffffff"
          strokeWidth={2}
          dot={{ r: 1.5 }}
          activeDot={{ r: 5 }}
          name="Mediana (baseline)"
          isAnimationActive={false}
        />
        {scenario ? (
          <Line
            type="monotone"
            dataKey="scenario_p50"
            stroke="#fa5252"
            strokeWidth={2}
            strokeDasharray="6 6"
            dot={{ r: 1.5 }}
            activeDot={{ r: 5 }}
            name="Mediana (escenario)"
            isAnimationActive={false}
          />
        ) : null}
      </AreaChart>
    </ResponsiveContainer>
  );
}
