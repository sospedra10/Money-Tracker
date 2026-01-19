import {
  AppShell,
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  NumberInput,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  Title,
} from "@mantine/core";
import { useEffect, useMemo, useState } from "react";

import {
  addHistoryEntry,
  fetchDashboard,
  fetchMeta,
  type DashboardRequest,
  type DashboardResponse,
  type Meta,
} from "./api/client";
import { formatCurrency, formatPct } from "./lib/format";
import { AllocationPie } from "./components/AllocationPie";
import { MonteCarloFanChart } from "./components/MonteCarloFanChart";
import { MonthlyChangeBar } from "./components/MonthlyChangeBar";
import { NetWorthChart } from "./components/NetWorthChart";

const DEFAULT_RATES_PCT: Record<string, number> = {
  "Remuneration Account": 2.5,
  "Real Estate": 11,
  "ETFs and Stocks": 6.5,
  Bank: 0,
  Crypto: 1,
  Reenlever: 11,
  Staking: 3,
  Others: 0,
};

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [updateCategory, setUpdateCategory] = useState<string | null>(null);
  const [updateAmount, setUpdateAmount] = useState<number | string>("");

  const [monthsToAnalyze, setMonthsToAnalyze] = useState<number>(10);
  const [projectionYears, setProjectionYears] = useState<number>(5);
  const [mcSimulations, setMcSimulations] = useState<number>(300);

  const [scenarioEnabled, setScenarioEnabled] = useState<boolean>(false);
  const [scenarioSavingsAdjust, setScenarioSavingsAdjust] = useState<number>(20);
  const [scenarioExtraMonthly, setScenarioExtraMonthly] = useState<number>(0);
  const [scenarioInterestShift, setScenarioInterestShift] = useState<number>(1);

  const [ratesPct, setRatesPct] = useState<Record<string, number>>(DEFAULT_RATES_PCT);

  const interestRates = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(ratesPct)) {
      out[k] = Math.max(v, 0) / 100;
    }
    return out;
  }, [ratesPct]);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const req: DashboardRequest = {
        interest_rates: interestRates,
        months_to_analyze: monthsToAnalyze,
        projection_years: projectionYears,
        mc_simulations: mcSimulations,
        scenario: {
          enabled: scenarioEnabled,
          savings_adjustment_pct: scenarioSavingsAdjust,
          extra_monthly: scenarioExtraMonthly,
          interest_shift_pp: scenarioInterestShift,
        },
      };
      const data = await fetchDashboard(req);
      setDashboard(data);
      if (data?.latest?.balances) {
        const firstCat = Object.keys(data.latest.balances)[0] ?? null;
        setUpdateCategory((prev) => prev ?? firstCat);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error cargando datos");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const m = await fetchMeta();
        setMeta(m);
        setUpdateCategory(m.categories[0] ?? null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Error cargando meta");
      }
    })();
  }, []);

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onAddEntry = async () => {
    if (!updateCategory) return;
    let amount: number | null = null;
    if (typeof updateAmount === "number") {
      amount = updateAmount;
    } else if (updateAmount.trim() !== "") {
      const parsed = Number(updateAmount);
      amount = Number.isFinite(parsed) ? parsed : null;
    }
    if (amount === null) return;
    setLoading(true);
    setError(null);
    try {
      await addHistoryEntry({ category: updateCategory, amount });
      setUpdateAmount("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error guardando entry");
    } finally {
      setLoading(false);
    }
  };

  const kpis = dashboard?.kpis;
  const latest = dashboard?.latest;
  const streak = dashboard?.monthly?.streak;
  const avgSavingsTotal = dashboard?.monthly?.avg_savings?.Total ?? 0;

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{ width: 360, breakpoint: "md" }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Title order={3}>Money Tracker</Title>
            <Badge variant="light">FastAPI + React</Badge>
          </Group>
          <Group>
            <Button variant="light" loading={loading} onClick={refresh}>
              Refresh
            </Button>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p={0}>
        <ScrollArea h="100%" offsetScrollbars scrollbarSize={8}>
          <Box p="md">
            <Stack gap="md">
              <Card withBorder>
                <Stack gap="xs">
                  <Title order={5}>Actualizar categoría</Title>
                  <Select
                    label="Categoría"
                    data={meta?.categories ?? []}
                    value={updateCategory}
                    onChange={setUpdateCategory}
                    searchable
                    nothingFoundMessage="No encontrada"
                  />
                  <NumberInput
                    label="Balance total (€)"
                    value={updateAmount}
                    onChange={setUpdateAmount}
                    min={0}
                    thousandSeparator=","
                    decimalScale={2}
                  />
                  <Button onClick={onAddEntry} loading={loading}>
                    Guardar
                  </Button>
                </Stack>
              </Card>

              <Card withBorder>
                <Stack gap="xs">
                  <Title order={5}>Parámetros</Title>
                  <NumberInput
                    label="Meses para media de ahorro"
                    value={monthsToAnalyze}
                    onChange={(v) =>
                      setMonthsToAnalyze(typeof v === "number" ? v : 10)
                    }
                    min={1}
                    max={60}
                  />
                  <NumberInput
                    label="Años de proyección"
                    value={projectionYears}
                    onChange={(v) =>
                      setProjectionYears(typeof v === "number" ? v : 5)
                    }
                    min={1}
                    max={60}
                  />
                  <NumberInput
                    label="Simulaciones Monte Carlo"
                    value={mcSimulations}
                    onChange={(v) =>
                      setMcSimulations(typeof v === "number" ? v : 300)
                    }
                    min={50}
                    max={5000}
                    step={50}
                  />
                </Stack>
              </Card>

              <Card withBorder>
                <Stack gap="xs">
                  <Group justify="space-between">
                    <Title order={5}>Escenario</Title>
                    <Switch
                      checked={scenarioEnabled}
                      onChange={(e) => setScenarioEnabled(e.currentTarget.checked)}
                      label="Activar"
                    />
                  </Group>
                  <NumberInput
                    label="Ajuste ahorro mensual (%)"
                    value={scenarioSavingsAdjust}
                    onChange={(v) =>
                      setScenarioSavingsAdjust(typeof v === "number" ? v : 20)
                    }
                    min={-50}
                    max={200}
                    step={5}
                  />
                  <NumberInput
                    label="Extra mensual (€)"
                    value={scenarioExtraMonthly}
                    onChange={(v) =>
                      setScenarioExtraMonthly(typeof v === "number" ? v : 0)
                    }
                    min={0}
                    step={50}
                  />
                  <NumberInput
                    label="Shift interés (puntos %)"
                    value={scenarioInterestShift}
                    onChange={(v) =>
                      setScenarioInterestShift(typeof v === "number" ? v : 1)
                    }
                    min={-5}
                    max={5}
                    step={0.1}
                  />
                </Stack>
              </Card>

              <Card withBorder>
                <Stack gap="xs">
                  <Title order={5}>Tasas (anual %)</Title>
                  {(meta?.categories ?? Object.keys(ratesPct)).map((cat) => (
                    <NumberInput
                      key={cat}
                      label={cat}
                      value={ratesPct[cat] ?? 0}
                      onChange={(v) =>
                        setRatesPct((prev) => ({
                          ...prev,
                          [cat]: typeof v === "number" ? v : prev[cat] ?? 0,
                        }))
                      }
                      min={0}
                      max={100}
                      step={0.1}
                    />
                  ))}
                </Stack>
              </Card>
            </Stack>
          </Box>
        </ScrollArea>
      </AppShell.Navbar>

      <AppShell.Main>
        <Stack gap="md">
          {error ? (
            <Card withBorder>
              <Text c="red">{error}</Text>
            </Card>
          ) : null}

          {!dashboard?.has_data ? (
            <Card withBorder>
              <Text>{dashboard?.message ?? "Sin datos"}</Text>
            </Card>
          ) : null}

          {dashboard?.has_data ? (
            <>
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 3, xl: 6 }}>
                <MetricCard
                  label="Total"
                  value={formatCurrency(latest?.total ?? 0)}
                  hint={latest?.as_of ? `As of ${latest.as_of.slice(0, 10)}` : undefined}
                />
                <MetricCard
                  label="Liquidez"
                  value={formatCurrency(kpis?.liquidity_total ?? 0)}
                  hint="Bank + Remuneration + Reenlever"
                />
                <MetricCard
                  label="Ahorro medio"
                  value={formatCurrency(avgSavingsTotal) + "/mes"}
                  hint={`Últimos ${monthsToAnalyze} meses`}
                />
                <MetricCard
                  label="Ingreso pasivo"
                  value={formatCurrency(kpis?.annual_passive_income ?? 0) + "/año"}
                  hint={formatCurrency((kpis?.annual_passive_income ?? 0) / 12) + "/mes"}
                />
                <MetricCard
                  label="CAGR"
                  value={formatPct(kpis?.cagr_total ?? 0) + "/año"}
                  hint={`Vol mensual: ${formatPct(kpis?.vol_monthly ?? 0)}`}
                />
                <MetricCard
                  label="Racha"
                  value={`${streak?.current ?? 0} meses`}
                  hint={`Mejor: ${streak?.best ?? 0}`}
                />
              </SimpleGrid>

              <SimpleGrid cols={{ base: 1, lg: 2 }}>
                <Card withBorder>
                  <Group justify="space-between" mb="xs">
                    <Title order={5}>Patrimonio (diario)</Title>
                    <Badge variant="light">
                      MDD {formatPct(kpis?.max_drawdown ?? 0)}
                    </Badge>
                  </Group>
                  <NetWorthChart data={dashboard.timeseries?.daily ?? []} />
                </Card>

                <Card withBorder>
                  <Group justify="space-between" mb="xs">
                    <Title order={5}>Asignación actual</Title>
                    <Badge variant="light">
                      HHI {(kpis?.hhi ?? 0).toFixed(3)}
                    </Badge>
                  </Group>
                  <AllocationPie balances={latest?.balances ?? {}} />
                </Card>
              </SimpleGrid>

              <SimpleGrid cols={{ base: 1, lg: 2 }}>
                <Card withBorder>
                  <Title order={5} mb="xs">
                    Cambios mensuales (Total)
                  </Title>
                  <MonthlyChangeBar changes={dashboard.monthly?.changes_long ?? []} />
                </Card>
              </SimpleGrid>

              <Card withBorder>
                <Group justify="space-between" mb="xs">
                  <Title order={5}>Monte Carlo (bandas)</Title>
                  <Badge variant="light">
                    {dashboard.projection?.years ?? 0} años
                  </Badge>
                </Group>
                <MonteCarloFanChart
                  baseline={dashboard.projection?.monte_carlo?.baseline ?? []}
                  scenario={dashboard.projection?.monte_carlo?.scenario ?? null}
                />
              </Card>

              {dashboard.insights?.length ? (
                <Card withBorder>
                  <Title order={5} mb="xs">
                    Insights
                  </Title>
                  <Divider mb="sm" />
                  <SimpleGrid cols={{ base: 1, md: 2 }}>
                    {dashboard.insights.map((insight) => (
                      <Box key={insight.title}>
                        <Text fw={600}>{insight.title}</Text>
                        <Text c="dimmed">{insight.text}</Text>
                      </Box>
                    ))}
                  </SimpleGrid>
                </Card>
              ) : null}
            </>
          ) : null}
        </Stack>
      </AppShell.Main>
    </AppShell>
  );
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <Card withBorder>
      <Stack gap={2}>
        <Text c="dimmed" size="sm">
          {label}
        </Text>
        <Text fw={700} size="xl">
          {value}
        </Text>
        {hint ? (
          <Text c="dimmed" size="xs">
            {hint}
          </Text>
        ) : null}
      </Stack>
    </Card>
  );
}
