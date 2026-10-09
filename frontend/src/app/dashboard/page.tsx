"use client";

import { useEffect, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, CheckCircle2, AlertTriangle, Zap } from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// --- Types ---

interface LatencyRow {
  step: string;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  sample_count: number;
}

interface PerPaperRow {
  paper_id: string;
  title: string;
  citation_count: number;
  avg_citation_accuracy: number;
}

interface MetricsSummary {
  components: {
    latency: { score: number; weight: number };
    empty_result_rate: { score: number; weight: number };
    citation_accuracy: { score: number; weight: number };
  };
  details: {
    latency_percentiles: LatencyRow[];
    empty_result_rate_24h: number;
    avg_citation_accuracy: number;
  };
  per_paper: PerPaperRow[];
}

interface TimeseriesPoint {
  date: string;
  span_counts: {
    total: number;
    embed: number;
    rewrite: number;
    search: number;
    lexical: number;
    rerank: number;
    generate: number;
  };
  avg_latency_ms: {
    embed: number;
    rewrite: number;
    search: number;
    lexical: number;
    rerank: number;
    generate: number;
  };
  token_usage: {
    input: number;
    output: number;
  };
  empty_search_count: number;
}

interface TimeseriesResponse {
  range_days: number;
  data: TimeseriesPoint[];
}

interface LowScoreRow {
  query: string;
  score: number;
  timestamp: string;
}

// --- helpers ---

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatTime(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

// --- Custom tooltip ---

function ChartTooltip({
  active,
  payload,
  label,
  valueFormatter,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
  valueFormatter?: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 text-sm shadow-sm">
      <p className="mb-1 font-medium text-foreground">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="text-muted-foreground text-xs">
          <span
            className="inline-block mr-1.5 size-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          {entry.name}: {valueFormatter ? valueFormatter(entry.value) : entry.value}
        </p>
      ))}
    </div>
  );
}

// --- Loading skeleton ---

function SkeletonCard() {
  return (
    <Card>
      <CardHeader>
        <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
      </CardHeader>
      <CardContent>
        <div className="h-48 animate-pulse rounded bg-muted" />
      </CardContent>
    </Card>
  );
}

// --- EmptyState ---

function EmptyState({ message }: { message: string }) {
  return (
    <Card>
      <CardContent className="flex items-center justify-center py-12">
        <p className="text-sm text-muted-foreground">{message}</p>
      </CardContent>
    </Card>
  );
}

// --- Error banner ---

function ErrorBanner({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <span>{message}</span>
      <button
        onClick={onRetry}
        className="ml-4 rounded-md border border-red-200 px-3 py-1 text-xs font-medium text-red-600 transition-colors hover:bg-red-100"
      >
        Retry
      </button>
    </div>
  );
}

// --- KPI Summary Stat Cards Grid ---

function SummaryStatsGrid({
  summary,
  timeseries7d,
  loading,
}: {
  summary: MetricsSummary | null;
  timeseries7d: TimeseriesResponse | null;
  loading: boolean;
}) {
  const avgCitationAccuracy = summary?.details?.avg_citation_accuracy ?? 0;
  const emptyRate24h = summary?.details?.empty_result_rate_24h ?? 0;

  const p50Gen = summary?.details?.latency_percentiles?.find((l) => l.step === "generate")?.p50_ms ?? 0;
  const p95Gen = summary?.details?.latency_percentiles?.find((l) => l.step === "generate")?.p95_ms ?? 0;

  const totalOps = useMemo(() => {
    if (!timeseries7d?.data) return 0;
    return timeseries7d.data.reduce((acc, p) => acc + (p.span_counts?.total || 0), 0);
  }, [timeseries7d]);

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardContent className="p-5">
              <div className="h-4 w-1/2 animate-pulse rounded bg-muted mb-3" />
              <div className="h-8 w-3/4 animate-pulse rounded bg-muted" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {/* Card 1: Generation Latency */}
      <Card className="relative overflow-hidden border-slate-200 shadow-xs hover:border-slate-300 transition-all">
        <CardContent className="p-5">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Pipeline Latency</span>
            <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
              <Clock className="size-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {p50Gen >= 1000 ? `${(p50Gen / 1000).toFixed(2)}s` : `${Math.round(p50Gen)}ms`}
            </span>
            <span className="text-xs text-slate-500 font-mono">p50</span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            p95: <span className="font-mono text-slate-700 font-semibold">{p95Gen >= 1000 ? `${(p95Gen / 1000).toFixed(2)}s` : `${Math.round(p95Gen)}ms`}</span>
          </p>
        </CardContent>
      </Card>

      {/* Card 2: Citation Accuracy */}
      <Card className="relative overflow-hidden border-slate-200 shadow-xs hover:border-slate-300 transition-all">
        <CardContent className="p-5">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Citation Accuracy</span>
            <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
              <CheckCircle2 className="size-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {(avgCitationAccuracy * 100).toFixed(1)}%
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Grounded citation ratio across responses
          </p>
        </CardContent>
      </Card>

      {/* Card 3: 24h Empty Result Rate */}
      <Card className="relative overflow-hidden border-slate-200 shadow-xs hover:border-slate-300 transition-all">
        <CardContent className="p-5">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">24h Empty Result Rate</span>
            <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
              <AlertTriangle className="size-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold tracking-tight font-mono ${emptyRate24h > 0.2 ? "text-amber-700" : "text-slate-900"}`}>
              {(emptyRate24h * 100).toFixed(1)}%
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Queries yielding 0 retrieved chunks
          </p>
        </CardContent>
      </Card>

      {/* Card 4: 7d Operations */}
      <Card className="relative overflow-hidden border-slate-200 shadow-xs hover:border-slate-300 transition-all">
        <CardContent className="p-5">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">7-Day Activity</span>
            <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
              <Zap className="size-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {totalOps.toLocaleString()}
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Total pipeline operations recorded
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// --- Faithfulness Trend ---

function computeQualityScore(point: TimeseriesPoint): number {
  const searchCount = point.span_counts.search;
  if (searchCount === 0) return 1;
  return Math.round((1 - point.empty_search_count / searchCount) * 1000) / 1000;
}

function FaithfulnessTrend({
  data7d,
  data30d,
  loading,
}: {
  data7d: TimeseriesPoint[];
  data30d: TimeseriesPoint[];
  loading: boolean;
}) {
  const chartData = useMemo(() => {
    const days = new Map<string, { date: string; score7d: number | null; score30d: number | null }>();

    for (const p of data7d) {
      const key = p.date.slice(0, 10);
      if (!days.has(key)) days.set(key, { date: key, score7d: null, score30d: null });
      days.get(key)!.score7d = computeQualityScore(p);
    }

    for (const p of data30d) {
      const key = p.date.slice(0, 10);
      if (!days.has(key)) days.set(key, { date: key, score7d: null, score30d: null });
      days.get(key)!.score30d = computeQualityScore(p);
    }

    return Array.from(days.values()).sort((a, b) => a.date.localeCompare(b.date));
  }, [data7d, data30d]);

  if (loading) return <SkeletonCard />;
  if (chartData.length === 0)
    return <EmptyState message="No data yet" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Search quality trend</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#bfbfbf" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fontSize: 11, fill: "#595959" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 11, fill: "#595959" }}
            />
            <Tooltip
              content={<ChartTooltip valueFormatter={(v) => (v * 100).toFixed(1) + "%"} />}
            />
            <Legend
              formatter={(value) => (
                <span className="text-xs text-muted-foreground">{value}</span>
              )}
            />
            <Line
              type="monotone"
              dataKey="score7d"
              name="7-day"
              stroke="#404040"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="score30d"
              name="30-day"
              stroke="#8c8c8c"
              strokeWidth={2}
              dot={false}
              connectNulls
              strokeDasharray="4 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// --- Per-paper citation accuracy ---

function PerPaperAccuracy({
  data,
  loading,
}: {
  data: PerPaperRow[];
  loading: boolean;
}) {
  const chartData = useMemo(
    () =>
      data
        .map((d) => ({
          title: d.title
            ? d.title.length > 35
              ? d.title.slice(0, 35) + "..."
              : d.title
            : `Paper ${d.paper_id.slice(0, 8)}`,
          accuracy: d.avg_citation_accuracy,
        }))
        .sort((a, b) => a.accuracy - b.accuracy),
    [data]
  );

  if (loading) return <SkeletonCard />;
  if (chartData.length === 0) return <EmptyState message="No data yet" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Citation accuracy per paper</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 48)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
            <CartesianGrid stroke="#bfbfbf" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 11, fill: "#595959" }}
            />
            <YAxis
              dataKey="title"
              type="category"
              tick={{ fontSize: 11, fill: "#404040" }}
              width={160}
            />
            <Tooltip
              content={<ChartTooltip valueFormatter={(v) => (v * 100).toFixed(1) + "%"} />}
            />
            <Bar
              dataKey="accuracy"
              fill="#595959"
              radius={[0, 3, 3, 0]}
              maxBarSize={16}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// --- Latency breakdown ---

function LatencyBreakdown({
  data,
  loading,
}: {
  data: LatencyRow[];
  loading: boolean;
}) {
  const chartData = useMemo(() => {
    const order = ["embed", "query_rewrite", "vector_search", "lexical_search", "rerank", "generate"];
    const items = data.filter((d) => order.includes(d.step));
    const sorted = items.sort(
      (a, b) => order.indexOf(a.step) - order.indexOf(b.step)
    );
    const stepLabel = (step: string) =>
      step === "vector_search" ? "search"
      : step === "query_rewrite" ? "rewrite"
      : step === "lexical_search" ? "lexical"
      : step;
    return sorted.map((d) => ({
      step: stepLabel(d.step),
      p50: d.p50_ms,
      p95: d.p95_ms,
    }));
  }, [data]);

  if (loading) return <SkeletonCard />;
  if (chartData.length === 0) return <EmptyState message="No data yet" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Latency by step</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} barGap={0} barCategoryGap="20%">
            <CartesianGrid stroke="#bfbfbf" strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 11, fill: "#595959" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#595959" }}
              tickFormatter={(v) => (v >= 1000 ? (v / 1000).toFixed(1) + "s" : Math.round(v) + "ms")}
            />
            <Tooltip
              content={<ChartTooltip valueFormatter={(v) => (v >= 1000 ? (v / 1000).toFixed(2) + "s" : Math.round(v) + "ms")} />}
            />
            <Legend
              formatter={(value) => (
                <span className="text-xs text-muted-foreground">{value}</span>
              )}
            />
            <Bar dataKey="p50" name="p50" fill="#737373" radius={[3, 3, 0, 0]} maxBarSize={28} />
            <Bar dataKey="p95" name="p95" fill="#404040" radius={[3, 3, 0, 0]} maxBarSize={28} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// --- Empty result rate sparkline ---

function EmptyResultRate({
  data,
  loading,
}: {
  data: TimeseriesPoint[];
  loading: boolean;
}) {
  const chartData = useMemo(() => {
    return data.map((p) => {
      const searchCount = p.span_counts.search;
      const rate = searchCount > 0 ? p.empty_search_count / searchCount : 0;
      return {
        date: p.date,
        rate: Math.round(rate * 10000) / 10000,
      };
    });
  }, [data]);

  if (loading) return <SkeletonCard />;
  if (chartData.length === 0) return <EmptyState message="No data yet" />;

  const latestRate = chartData.length > 0 ? chartData[chartData.length - 1].rate : 0;
  const threshold = 0.05;
  const isAboveThreshold = latestRate >= threshold;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Empty result rate trend</CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Latest 24h:</span>
            <span
              className={`text-xs font-mono font-semibold ${
                isAboveThreshold ? "text-amber-700" : "text-slate-900"
              }`}
            >
              {(latestRate * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={chartData}>
            <Area
              type="monotone"
              dataKey="rate"
              stroke="#404040"
              fill="#e5e5e5"
              strokeWidth={2}
              dot={false}
            />
            <Tooltip
              content={
                <ChartTooltip
                  valueFormatter={(v) => (v * 100).toFixed(1) + "%"}
                />
              }
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// --- Low-scoring queries table ---

function LowScoringQueries({
  data,
  loading,
}: {
  data: LowScoreRow[];
  loading: boolean;
}) {
  if (loading)
    return (
      <Card>
        <CardHeader>
          <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
        </CardHeader>
        <CardContent>
          <div className="h-32 animate-pulse rounded bg-muted" />
        </CardContent>
      </Card>
    );

  if (data.length === 0) return <EmptyState message="No data yet" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent low-scoring queries</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="pb-2 pr-4 font-medium">Score</th>
                <th className="pb-2 pr-4 font-medium">Query</th>
                <th className="pb-2 font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr
                  key={i}
                  className="border-b border-border last:border-0"
                >
                  <td className="py-2.5 pr-4">
                    <span
                      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium font-mono ${
                        row.score < 0.6
                          ? "bg-red-50 text-red-700 border border-red-200"
                          : row.score < 0.8
                            ? "bg-amber-50 text-amber-700 border border-amber-200"
                            : "bg-slate-100 text-slate-700 border border-slate-200"
                      }`}
                    >
                      {(row.score * 100).toFixed(0)}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-muted-foreground max-w-[250px] truncate">
                    {row.query.length > 50
                      ? row.query.slice(0, 50) + "..."
                      : row.query}
                  </td>
                  <td className="py-2.5 whitespace-nowrap text-muted-foreground">
                    {formatTime(row.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// --- Pipeline activity stacked area ---

function PipelineActivity({
  data,
  loading,
}: {
  data: TimeseriesPoint[];
  loading: boolean;
}) {
  const chartData = useMemo(
    () =>
      data.map((p) => ({
        date: p.date.slice(0, 10),
        Embeddings: p.span_counts.embed,
        Rewrite: p.span_counts.rewrite,
        Search: p.span_counts.search,
        Lexical: p.span_counts.lexical,
        Rerank: p.span_counts.rerank,
        Generate: p.span_counts.generate,
      })),
    [data]
  );

  if (loading) return <SkeletonCard />;
  if (chartData.length === 0) return <EmptyState message="No data yet" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Pipeline activity</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={chartData}>
            <CartesianGrid stroke="#d9d9d9" strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#595959" }} />
            <YAxis
              allowDecimals={false}
              tick={{ fontSize: 10, fill: "#595959" }}
            />
            <Tooltip />
            <Legend
              formatter={(value) => (
                <span className="text-xs text-muted-foreground">{value}</span>
              )}
            />
            <Area
              type="monotone"
              dataKey="Embeddings"
              stackId="1"
              stroke="#737373"
              fill="#d9d9d9"
            />
            <Area
              type="monotone"
              dataKey="Rewrite"
              stackId="1"
              stroke="#a3a3a3"
              fill="#e5e5e5"
            />
            <Area
              type="monotone"
              dataKey="Search"
              stackId="1"
              stroke="#404040"
              fill="#a3a3a3"
            />
            <Area
              type="monotone"
              dataKey="Lexical"
              stackId="1"
              stroke="#595959"
              fill="#d4d4d4"
            />
            <Area
              type="monotone"
              dataKey="Rerank"
              stackId="1"
              stroke="#262626"
              fill="#737373"
            />
            <Area
              type="monotone"
              dataKey="Generate"
              stackId="1"
              stroke="#171717"
              fill="#404040"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// --- Main page ---

export default function DashboardPage() {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [timeseries7d, setTimeseries7d] = useState<TimeseriesResponse | null>(
    null
  );
  const [timeseries30d, setTimeseries30d] =
    useState<TimeseriesResponse | null>(null);
  const [lowScoreData, setLowScoreData] = useState<LowScoreRow[]>([]);
  const [rerankerStats, setRerankerStats] = useState<any>(null);
  const [tokenUsage, setTokenUsage] = useState<any>(null);
  const [surfacedTraces, setSurfacedTraces] = useState<any>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [summaryRes, ts7dRes, ts30dRes, rerankRes, tokenRes, surfacedRes] = await Promise.all([
        fetch(`${API_BASE}/api/metrics/summary`),
        fetch(`${API_BASE}/api/metrics/timeseries?range=7d`),
        fetch(`${API_BASE}/api/metrics/timeseries?range=30d`),
        fetch(`${API_BASE}/api/metrics/reranker-stats?range=7d`),
        fetch(`${API_BASE}/api/metrics/token-usage?range=7d`),
        fetch(`${API_BASE}/api/metrics/surfaced-traces?limit=5`),
      ]);

      if (!summaryRes.ok) throw new Error("Failed to fetch metrics summary");
      if (!ts7dRes.ok) throw new Error("Failed to fetch 7-day timeseries");
      if (!ts30dRes.ok) throw new Error("Failed to fetch 30-day timeseries");

      const summaryData: MetricsSummary = await summaryRes.json();
      const ts7dData: TimeseriesResponse = await ts7dRes.json();
      const ts30dData: TimeseriesResponse = await ts30dRes.json();

      setSummary(summaryData);
      setTimeseries7d(ts7dData);
      setTimeseries30d(ts30dData);

      if (rerankRes.ok) setRerankerStats(await rerankRes.json());
      if (tokenRes.ok) setTokenUsage(await tokenRes.json());
      if (surfacedRes.ok) setSurfacedTraces(await surfacedRes.json());

      // Try fetching low-scoring queries
      try {
        const lsqRes = await fetch(
          `${API_BASE}/api/metrics/low-scoring-queries`
        );
        if (lsqRes.ok) {
          const lsqData: LowScoreRow[] = await lsqRes.json();
          setLowScoreData(lsqData);
        }
      } catch {
        // endpoint not available yet
      }
    } catch (err) {
      // Surface a generic message: raw fetch errors can embed backend
      // hostnames and DNS details (e.g. DNS_HOSTNAME_NOT_FOUND).
      console.error(err);
      setError("Failed to load metrics. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const summaryLoading = loading && !summary;
  const ts7dLoading = loading && !timeseries7d;
  const ts30dLoading = loading && !timeseries30d;
  const hasAnyData =
    summary || timeseries7d || timeseries30d || lowScoreData.length > 0;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground font-serif tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          System operational metrics and RAG pipeline telemetry
        </p>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-6">
          <ErrorBanner message={error} onRetry={fetchData} />
        </div>
      )}

      {/* Empty state -- no data at all */}
      {!loading && !error && !hasAnyData && (
        <EmptyState message="No data yet" />
      )}

      {/* Main Grid sections */}
      <div className="space-y-6">
        {/* Row 1: KPI Stat Cards Grid */}
        <SummaryStatsGrid
          summary={summary}
          timeseries7d={timeseries7d}
          loading={summaryLoading}
        />

        {/* Row 2: Search Quality Trend & Pipeline Activity */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <FaithfulnessTrend
            data7d={timeseries7d?.data || []}
            data30d={timeseries30d?.data || []}
            loading={ts7dLoading && ts30dLoading}
          />
          <PipelineActivity data={timeseries7d?.data || []} loading={ts7dLoading} />
        </div>

        {/* Row 3: Per-paper accuracy + Latency breakdown */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <PerPaperAccuracy
            data={summary?.per_paper || []}
            loading={summaryLoading}
          />
          <LatencyBreakdown
            data={summary?.details.latency_percentiles || []}
            loading={summaryLoading}
          />
        </div>

        {/* Row 4: Reranker Fallback Rate & Token Cost Trend (Dark Slate Theme) */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Reranker Fallback Rate */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Reranker Fallback Rate</CardTitle>
              {rerankerStats && (
                <Badge variant="outline" className="font-mono text-xs border-slate-200 text-slate-700 bg-slate-50">
                  Fallback: {rerankerStats.fallback_rate_pct}%
                </Badge>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              {rerankerStats?.daily?.length > 0 ? (
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={rerankerStats.daily}>
                    <CartesianGrid stroke="#bfbfbf" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#595959" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#595959" }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="total_calls" name="Total Reranks" fill="#404040" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="fallback_calls" name="Fallbacks" fill="#8c8c8c" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-muted-foreground py-4 text-center">No reranker stats recorded yet.</p>
              )}
            </CardContent>
          </Card>

          {/* Token Usage & Cost Trend */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Token Cost & Usage Trend</CardTitle>
              {tokenUsage && (
                <Badge variant="outline" className="font-mono text-xs border-slate-200 text-slate-700 bg-slate-50">
                  {tokenUsage.total_tokens?.toLocaleString() || 0} Total Tokens
                </Badge>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              {tokenUsage?.daily?.length > 0 ? (
                <ResponsiveContainer width="100%" height={160}>
                  <AreaChart data={tokenUsage.daily}>
                    <CartesianGrid stroke="#bfbfbf" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#595959" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#595959" }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Area type="monotone" dataKey="input_tokens" name="Input Tokens" stroke="#404040" fill="#d4d4d4" />
                    <Area type="monotone" dataKey="output_tokens" name="Output Tokens" stroke="#737373" fill="#e5e5e5" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-muted-foreground py-4 text-center">No token usage metrics available.</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Row 5: Empty Result Rate Trend */}
        <EmptyResultRate
          data={timeseries7d?.data || []}
          loading={ts7dLoading}
        />

        {/* Row 6: Surfaced Issues & Slow Traces Panel */}
        {surfacedTraces && (surfacedTraces.recent_errors?.length > 0 || surfacedTraces.slow_traces?.length > 0) && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Surfaced Pipeline Issues & Slow Traces</CardTitle>
              <a href="/traces" className="text-xs font-semibold text-slate-700 hover:text-slate-900 underline">
                Open Trace Inspector →
              </a>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Recent Errors */}
                <div className="space-y-2">
                  <span className="font-bold text-slate-700 tracking-wide uppercase text-[10px]">
                    Recent Errors & Fallbacks ({surfacedTraces.recent_errors?.length || 0})
                  </span>
                  <div className="space-y-1.5">
                    {surfacedTraces.recent_errors?.map((tr: any) => (
                      <div key={tr.trace_id} className="p-2 rounded bg-slate-50 border border-slate-200 flex items-center justify-between">
                        <span className="font-mono font-semibold truncate max-w-[180px]">Trace {tr.trace_id.slice(0, 8)}...</span>
                        <Badge variant="outline" className="text-[10px] border-amber-300 text-amber-700 bg-amber-50">
                          {tr.has_error ? "Error" : "Rerank Fallback"}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Slow Traces */}
                <div className="space-y-2">
                  <span className="font-bold text-slate-700 tracking-wide uppercase text-[10px]">
                    Slow Traces (p95+ &gt; 2s) ({surfacedTraces.slow_traces?.length || 0})
                  </span>
                  <div className="space-y-1.5">
                    {surfacedTraces.slow_traces?.map((tr: any) => (
                      <div key={tr.trace_id} className="p-2 rounded bg-slate-50 border border-slate-200 flex items-center justify-between">
                        <span className="font-mono font-semibold truncate max-w-[180px]">Trace {tr.trace_id.slice(0, 8)}...</span>
                        <span className="font-mono font-bold text-slate-700">{tr.total_duration_ms}ms</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Row 7: Low-scoring queries */}
        <LowScoringQueries
          data={lowScoreData}
          loading={loading && lowScoreData.length === 0}
        />
      </div>
    </div>
  );
}
