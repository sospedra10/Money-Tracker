import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency } from "../lib/format";
import { CATEGORY_COLORS } from "./colors";

export function NetWorthChart({
  data,
}: {
  data: Array<Record<string, string | number>>;
}) {
  if (!data.length) return null;

  const keys = Object.keys(data[0] ?? {}).filter(
    (k) => k !== "date" && k !== "Total",
  );

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
        <Tooltip
          formatter={(v) => formatCurrency(Number(v))}
          labelFormatter={(l) => `Fecha: ${l}`}
        />
        <Legend />
        {keys.map((k) => (
          <Area
            key={k}
            type="monotone"
            dataKey={k}
            stackId="1"
            stroke={CATEGORY_COLORS[k] ?? "#999"}
            fill={CATEGORY_COLORS[k] ?? "#999"}
            fillOpacity={0.22}
            dot={false}
          />
        ))}
        <Area
          type="monotone"
          dataKey="Total"
          stroke="#ffffff"
          fill="#ffffff"
          fillOpacity={0.06}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

