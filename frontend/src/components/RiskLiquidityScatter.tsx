import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { formatCurrency, formatPct } from "../lib/format";
import { CATEGORY_COLORS } from "./colors";

type Point = {
  category: string;
  balance: number;
  risk: number;
  liquidity: number;
  expected_return: number;
};

export function RiskLiquidityScatter({ points }: { points: Point[] }) {
  if (!points.length) return null;

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
        <XAxis
          type="number"
          dataKey="liquidity"
          domain={[1, 5]}
          tickCount={5}
          name="Liquidez"
        />
        <YAxis
          type="number"
          dataKey="risk"
          domain={[1, 5]}
          tickCount={5}
          name="Riesgo"
        />
        <ZAxis type="number" dataKey="balance" range={[80, 600]} name="Balance" />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(value, name, props) => {
            const p = props.payload as Point;
            if (name === "balance") return formatCurrency(Number(value));
            if (name === "expected_return") return formatPct(Number(value));
            return String(value);
          }}
          labelFormatter={(_, payload) => {
            const p = payload?.[0]?.payload as Point | undefined;
            return p?.category ? `Categoría: ${p.category}` : "";
          }}
        />
        <Scatter
          data={points}
          fill="#4dabf7"
          shape={(props: any) => {
            const { cx, cy, payload, size } = props as any;
            const p = payload as Point;
            const r = Math.sqrt(Math.max(size ?? 0, 0)) / 2;
            return (
              <circle
                cx={cx}
                cy={cy}
                r={Math.max(6, r)}
                fill={CATEGORY_COLORS[p.category] ?? "#4dabf7"}
                fillOpacity={0.75}
                stroke="#111"
                strokeOpacity={0.35}
              />
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
