import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency } from "../lib/format";

export function MonthlyChangeBar({
  changes,
}: {
  changes: Array<{ month: string; category: string; change: number }>;
}) {
  const rows = changes
    .filter((r) => r.category === "Total")
    .slice(-18)
    .map((r) => ({ month: r.month, change: r.change }));

  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={rows}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
        <XAxis dataKey="month" tick={{ fontSize: 12 }} />
        <YAxis
          tick={{ fontSize: 12 }}
          tickFormatter={(v) => formatCurrency(Number(v))}
          width={90}
        />
        <Tooltip
          formatter={(v) => formatCurrency(Number(v))}
          labelFormatter={(l) => `Mes: ${l}`}
        />
        <ReferenceLine y={0} stroke="#666" opacity={0.6} />
        <Bar dataKey="change" fill="#4dabf7" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

