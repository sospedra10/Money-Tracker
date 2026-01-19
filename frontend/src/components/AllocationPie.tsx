import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { formatCurrency } from "../lib/format";
import { CATEGORY_COLORS } from "./colors";

export function AllocationPie({ balances }: { balances: Record<string, number> }) {
  const rows = Object.entries(balances)
    .map(([name, value]) => ({ name, value }))
    .filter((r) => r.value > 0)
    .sort((a, b) => b.value - a.value);

  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={360}>
      <PieChart>
        <Pie
          data={rows}
          dataKey="value"
          nameKey="name"
          innerRadius={70}
          outerRadius={120}
          paddingAngle={2}
        >
          {rows.map((entry) => (
            <Cell
              key={entry.name}
              fill={CATEGORY_COLORS[entry.name] ?? "#999"}
            />
          ))}
        </Pie>
        <Tooltip formatter={(v) => formatCurrency(Number(v))} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

