import { ScrollArea, Table } from "@mantine/core";

type Correlation = { categories: string[]; matrix: number[][] } | undefined;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function colorForValue(value: number): string {
  const v = Math.max(-1, Math.min(1, value));
  if (v === 0) return "rgba(255,255,255,0.06)";

  // Diverging: red (-1) -> neutral (0) -> blue (+1)
  if (v < 0) {
    const t = Math.abs(v);
    const r = 250;
    const g = Math.round(lerp(82, 255, 1 - t));
    const b = Math.round(lerp(82, 255, 1 - t));
    return `rgba(${r},${g},${b},0.55)`;
  }
  const t = v;
  const r = Math.round(lerp(255, 77, t));
  const g = Math.round(lerp(255, 171, t));
  const b = 247;
  return `rgba(${r},${g},${b},0.55)`;
}

export function CorrelationHeatmap({ correlation }: { correlation: Correlation }) {
  const cats = correlation?.categories ?? [];
  const m = correlation?.matrix ?? [];
  if (!cats.length || !m.length) return null;

  return (
    <ScrollArea>
      <Table withColumnBorders withTableBorder striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th></Table.Th>
            {cats.map((c) => (
              <Table.Th key={c} style={{ whiteSpace: "nowrap" }}>
                {c}
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {cats.map((rowCat, i) => (
            <Table.Tr key={rowCat}>
              <Table.Th style={{ whiteSpace: "nowrap" }}>{rowCat}</Table.Th>
              {cats.map((colCat, j) => {
                const value = m[i]?.[j] ?? 0;
                return (
                  <Table.Td
                    key={`${rowCat}-${colCat}`}
                    style={{
                      textAlign: "right",
                      background: colorForValue(value),
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {value.toFixed(2)}
                  </Table.Td>
                );
              })}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}

