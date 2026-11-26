"use client";

import { useEffect, useRef, useState } from "react";

type Pred = { label: string; prob: number };
type Result = {
  current: {
    label: string;
    confidence: number;
    uncertain: boolean;
    why: string;
    top3: Pred[];
  };
  gradcam: string;
  next: Pred[] | null;
  coaching: string | null;
  history: string[];
  history_ids: number[];
  error?: string;
};

const DEFAULT_API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const STROKES = ["Short Serve", "Lift", "Clear", "Drop Shot", "Smash"];

// one signature colour per stroke — used on the timeline, bars and footer
const CLASS_COLORS: Record<string, string> = {
  "Short Serve": "#0ea5e9",
  Lift: "#f59e0b",
  Clear: "#10b981",
  "Drop Shot": "#8b5cf6",
  Smash: "#e11d48",
};

const pct = (p: number) => Math.round(p * 100);

function tone(conf: number) {
  if (conf >= 0.75)
    return { color: "#10b981", chip: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-600/20", label: "High confidence" };
  if (conf >= 0.5)
    return { color: "#d97706", chip: "bg-amber-50 text-amber-700 ring-1 ring-amber-600/20", label: "Moderate confidence" };
  return { color: "#e11d48", chip: "bg-rose-50 text-rose-700 ring-1 ring-rose-600/20", label: "Low confidence" };
}

/* ---------------- decorative + small pieces ---------------- */

// top-down badminton court line art (true proportions, 13.4m x 6.1m)
function CourtLines({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 134 61" fill="none" className={className} aria-hidden>
      <g stroke="currentColor" strokeWidth="0.7">
        <rect x="0.4" y="0.4" width="133.2" height="60.2" />
        <line x1="67" y1="0" x2="67" y2="61" strokeDasharray="2.4 1.8" strokeWidth="0.9" />
        <line x1="47.2" y1="0" x2="47.2" y2="61" />
        <line x1="86.8" y1="0" x2="86.8" y2="61" />
        <line x1="7.6" y1="0" x2="7.6" y2="61" />
        <line x1="126.4" y1="0" x2="126.4" y2="61" />
        <line x1="0" y1="4.6" x2="134" y2="4.6" />
        <line x1="0" y1="56.4" x2="134" y2="56.4" />
        <line x1="0" y1="30.5" x2="47.2" y2="30.5" />
        <line x1="86.8" y1="30.5" x2="134" y2="30.5" />
      </g>
    </svg>
  );
}

function ShuttleMark({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="17.5" r="3" fill="currentColor" stroke="none" />
      <path d="M12 14.5 L6.5 4 M12 14.5 L12 3 M12 14.5 L17.5 4" />
    </svg>
  );
}

function Ring({ value }: { value: number }) {
  const r = 36;
  const C = 2 * Math.PI * r;
  const t = tone(value);
  return (
    <div className="relative h-[100px] w-[100px] shrink-0">
      <svg viewBox="0 0 92 92" className="h-full w-full -rotate-90">
        <circle cx="46" cy="46" r={r} fill="none" stroke="#edf2ef" strokeWidth="9" />
        <circle
          cx="46"
          cy="46"
          r={r}
          fill="none"
          stroke={t.color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - value)}
          style={{ transition: "stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="tnum text-2xl font-bold leading-none text-ink">{pct(value)}%</span>
        <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-wider text-sub">confidence</span>
      </div>
    </div>
  );
}

function Bars({ items, accent }: { items: Pred[]; accent?: string }) {
  return (
    <div className="space-y-3">
      {items.map((it, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="w-24 shrink-0 text-sm font-medium text-sub">{it.label}</span>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full"
              style={{
                width: `${pct(it.prob)}%`,
                background: i === 0 ? accent || CLASS_COLORS[it.label] || "#10b981" : "#cbd5e1",
                transition: "width .7s ease",
              }}
            />
          </div>
          <span className="tnum w-10 shrink-0 text-right text-sm font-semibold text-ink">{pct(it.prob)}%</span>
        </div>
      ))}
    </div>
  );
}

function Card({
  title,
  children,
  className = "",
  delay = 0,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <section
      className={"fade-up rounded-2xl border border-line bg-surface p-6 shadow-card " + className}
      style={{ animationDelay: `${delay}ms` }}
    >
      {title && (
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.16em] text-sub">{title}</h2>
      )}
      {children}
    </section>
  );
}

function StatChip({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex items-baseline gap-2 rounded-full border border-white/15 bg-white/[0.08] px-4 py-2 backdrop-blur">
      <span className="tnum font-display text-base font-extrabold text-white">{value}</span>
      <span className="text-xs font-medium text-white/70">{label}</span>
    </div>
  );
}

function RallyTimeline({ names }: { names: string[] }) {
  if (!names.length) return <p className="text-sm text-sub">No shots yet — upload the first clip of the rally.</p>;
  return (
    <div className="overflow-x-auto pb-1">
      <div className="flex items-start pt-1">
        {names.map((n, i) => {
          const last = i === names.length - 1;
          const color = CLASS_COLORS[n] || "#10b981";
          return (
            <div key={i} className="flex items-start">
              {i > 0 && <div className="mt-[15px] h-px w-9 shrink-0 bg-slate-300" />}
              <div className="flex w-[76px] shrink-0 flex-col items-center gap-1.5">
                <div
                  className="grid h-8 w-8 place-items-center rounded-full text-xs font-bold"
                  style={
                    last
                      ? { background: color, color: "#fff", boxShadow: `0 6px 16px ${color}55` }
                      : { background: "#fff", border: `2px solid ${color}`, color: "#0f172a" }
                  }
                >
                  {i + 1}
                </div>
                <span className={"text-center text-[11px] leading-tight " + (last ? "font-semibold text-ink" : "text-sub")}>
                  {n}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---------------- page ---------------- */

export default function Page() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [health, setHealth] = useState<"unknown" | "ok" | "down">("unknown");
  const [showSettings, setShowSettings] = useState(false);
  const [historyIds, setHistoryIds] = useState<number[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem("rallylens_api");
    if (saved) setApiUrl(saved);
    checkHealth(saved || DEFAULT_API);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function checkHealth(url = apiUrl) {
    try {
      const r = await fetch(url.replace(/\/$/, "") + "/health");
      const j = await r.json();
      setHealth(j.video_model_loaded ? "ok" : "down");
    } catch {
      setHealth("down");
    }
  }

  function saveApi() {
    localStorage.setItem("rallylens_api", apiUrl);
    checkHealth();
    setShowSettings(false);
  }

  async function analyze(file: File) {
    setLoading(true);
    setError(null);
    setFileName(file.name);
    setVideoUrl(URL.createObjectURL(file));
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("history", JSON.stringify(historyIds));
      const r = await fetch(apiUrl.replace(/\/$/, "") + "/analyze", { method: "POST", body: fd });
      const j: Result = await r.json();
      if (j.error) {
        setError(j.error);
      } else {
        setResult(j);
        setHistoryIds(j.history_ids);
        setHealth("ok");
      }
    } catch {
      setError("Could not reach the backend. Open Settings and check the API URL is correct and the server is running.");
      setHealth("down");
    } finally {
      setLoading(false);
    }
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) analyze(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) analyze(f);
  }

  function resetRally() {
    setHistoryIds([]);
    setResult(null);
    setVideoUrl(null);
    setFileName(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  const cur = result?.current;
  const t = cur ? tone(cur.confidence) : null;

  return (
    <div className="min-h-screen pb-16">
      {/* ============ hero band ============ */}
      <div className="relative overflow-hidden bg-hero text-white">
        <CourtLines className="pointer-events-none absolute -right-12 top-10 hidden w-[480px] text-white opacity-[0.16] lg:block" />

        {/* nav */}
        <nav className="relative mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/10 ring-1 ring-white/20">
              <ShuttleMark className="text-white" />
            </div>
            <div>
              <div className="font-display text-lg font-extrabold leading-none">RallyLens</div>
              <div className="mt-0.5 text-[11px] text-white/60">Badminton Stroke &amp; Rally Analytics</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-xs font-semibold ring-1 ring-white/15">
              <span
                className={
                  "h-2 w-2 rounded-full " +
                  (health === "ok" ? "bg-emerald-400" : health === "down" ? "bg-rose-400" : "bg-slate-300")
                }
              />
              {health === "ok" ? "Model online" : health === "down" ? "Offline" : "Checking…"}
            </span>
            <button
              onClick={() => setShowSettings((s) => !s)}
              className="rounded-lg px-3 py-1.5 text-xs font-semibold text-white/80 ring-1 ring-white/20 transition hover:bg-white/10"
            >
              Settings
            </button>
          </div>
        </nav>

        {/* settings */}
        {showSettings && (
          <div className="relative border-y border-white/10 bg-white/[0.06] backdrop-blur">
            <div className="mx-auto flex max-w-6xl flex-col gap-2 px-5 py-3 sm:flex-row sm:items-center">
              <label className="text-xs font-semibold text-white/70">Backend API URL</label>
              <input
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="https://your-backend.trycloudflare.com"
                className="flex-1 rounded-lg border border-white/15 bg-white/95 px-3 py-2 text-sm text-ink outline-none focus:border-white"
              />
              <button
                onClick={saveApi}
                className="rounded-lg bg-white px-4 py-2 text-sm font-bold text-court transition hover:bg-emerald-50"
              >
                Save &amp; connect
              </button>
            </div>
          </div>
        )}

        {/* hero copy */}
        <div className="relative mx-auto max-w-6xl px-5 pb-24 pt-10">
          <span className="inline-block rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-emerald-300">
            AI Badminton Analytics
          </span>
          <h1 className="font-display mt-4 max-w-xl text-4xl font-extrabold leading-[1.1] tracking-tight md:text-5xl">
            Every stroke, explained.
          </h1>
          <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-white/70">
            Upload a single clip — RallyLens identifies the stroke, shows <em>why</em> the model decided,
            forecasts the likely next shot, and gives a coaching cue. Built on two trained deep-learning
            models served in real time.
          </p>
          <div className="mt-6 flex flex-wrap gap-2.5">
            <StatChip value="95%" label="stroke accuracy" />
            <StatChip value="97%" label="top-3 next shot" />
            <StatChip value="449 fps" label="on CPU (INT8)" />
            <StatChip value="0.024" label="calibration error" />
          </div>
        </div>
      </div>

      {/* ============ content ============ */}
      <main className="mx-auto max-w-6xl px-5">
        {/* upload card overlapping the hero */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className={
            "relative z-10 -mt-14 rounded-2xl border-2 border-dashed bg-surface px-6 py-9 text-center shadow-lift transition " +
            (drag ? "border-brand bg-brand-soft" : "border-slate-300")
          }
        >
          <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-full bg-brand-soft text-brand">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
            </svg>
          </div>
          <p className="text-[15px] font-semibold text-ink">Drag &amp; drop a shot clip</p>
          <p className="mb-4 mt-0.5 text-sm text-sub">one stroke per clip — MP4 · MOV · AVI · MKV</p>
          <button
            onClick={() => fileRef.current?.click()}
            className="rounded-xl bg-brand px-6 py-2.5 text-sm font-bold text-white shadow-pop transition hover:bg-brand-dark"
          >
            Choose a clip
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
            onChange={onFile}
            className="hidden"
          />
          {fileName && <p className="mt-3 text-xs font-medium text-brand-dark">Selected: {fileName}</p>}
        </div>

        {/* states */}
        <div className="mt-6 space-y-5">
          {loading && (
            <Card>
              <div className="flex items-center justify-center gap-3 py-5 text-sm text-sub">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-brand" />
                <span>
                  Analyzing — sampling <b className="text-ink">16 frames</b> → ResNet-18 → Grad-CAM…
                </span>
              </div>
            </Card>
          )}

          {error && (
            <div className="fade-up rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm font-medium text-rose-700">
              {error}
            </div>
          )}

          {!loading && !result && !error && (
            <Card title="How it works">
              <ol className="grid gap-4 sm:grid-cols-3">
                {[
                  ["1", "Upload a clip", "A short, single-stroke clip from any match or training session."],
                  ["2", "Read the analysis", "Stroke + confidence, the Grad-CAM evidence, and the forecast next shot."],
                  ["3", "Build the rally", "Keep uploading shots — the timeline and coaching adapt to the rally."],
                ].map(([n, h, d]) => (
                  <li key={n} className="rounded-xl border border-line bg-page p-4">
                    <div className="mb-2 grid h-7 w-7 place-items-center rounded-full bg-court text-xs font-bold text-white">
                      {n}
                    </div>
                    <div className="text-sm font-semibold text-ink">{h}</div>
                    <div className="mt-1 text-[13px] leading-relaxed text-sub">{d}</div>
                  </li>
                ))}
              </ol>
            </Card>
          )}

          {/* ============ results ============ */}
          {!loading && result && cur && t && (
            <>
              {/* 01 — detected stroke */}
              <Card title="01 — Detected stroke" delay={0}>
                <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <div className="flex items-center gap-2.5">
                      <span
                        className="h-3 w-3 rounded-full"
                        style={{ background: CLASS_COLORS[cur.label] || "#10b981" }}
                      />
                      <span className="font-display text-4xl font-extrabold tracking-tight text-ink">
                        {cur.label}
                      </span>
                    </div>
                    <span className={"mt-3 inline-block rounded-full px-3 py-1 text-xs font-semibold " + t.chip}>
                      {t.label}
                    </span>
                    <p className="mt-3 max-w-md text-sm leading-relaxed text-sub">{cur.why}</p>
                  </div>
                  <Ring value={cur.confidence} />
                </div>

                <div className="mt-6 border-t border-line pt-5">
                  <div className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-sub">
                    Class probabilities
                  </div>
                  <Bars items={cur.top3} />
                </div>

                {cur.uncertain && (
                  <div className="mt-5 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" className="mt-0.5 shrink-0">
                      <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    Low confidence — this prediction is flagged for human review.
                  </div>
                )}
              </Card>

              {/* 02 — evidence */}
              <div className="grid gap-5 lg:grid-cols-2">
                <Card title="02 — Why · Grad-CAM evidence" delay={60}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={result.gradcam} alt="Grad-CAM heatmap" className="w-full rounded-xl border border-line" />
                  <p className="mt-3 text-[13px] leading-relaxed text-sub">
                    Warm regions show where the model looked when making this call — evidence it reads the
                    player and the court action, not the background.
                  </p>
                </Card>

                {videoUrl && (
                  <Card title="Your clip" delay={120}>
                    <video src={videoUrl} controls className="w-full rounded-xl border border-line bg-black" />
                  </Card>
                )}
              </div>

              {/* 03 — tactics */}
              <div className="grid gap-5 lg:grid-cols-2">
                {result.next && (
                  <Card title="03 — Predicted next shot" delay={180}>
                    <Bars items={result.next} accent="#2563eb" />
                    <p className="mt-3 text-[13px] text-sub">
                      Forecast by the sequence model from the last shots of the rally.
                    </p>
                  </Card>
                )}

                <Card title="04 — Rally timeline" delay={240}>
                  <RallyTimeline names={result.history} />
                  <button
                    onClick={resetRally}
                    className="mt-4 rounded-lg border border-line bg-white px-3.5 py-2 text-xs font-semibold text-sub transition hover:bg-page"
                  >
                    Reset rally
                  </button>
                </Card>
              </div>

              {/* coaching */}
              {result.coaching && (
                <div
                  className="fade-up flex items-start gap-3 rounded-2xl border border-brand/20 bg-brand-soft px-5 py-4"
                  style={{ animationDelay: "300ms" }}
                >
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-court text-white">
                    <ShuttleMark className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-xs font-bold uppercase tracking-[0.16em] text-brand-dark">Coach&apos;s cue</div>
                    <p className="mt-1 text-[15px] leading-relaxed text-ink">{result.coaching}</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* footer */}
        <footer className="mt-12 border-t border-line pt-6 text-center text-xs text-sub">
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2">
            {STROKES.map((s) => (
              <span key={s} className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: CLASS_COLORS[s] }} />
                {s}
              </span>
            ))}
          </div>
          <p className="mt-2 text-slate-400">
            RallyLens — temporal-pooling ResNet-18 stroke classifier + GRU next-shot model, served via FastAPI
          </p>
        </footer>
      </main>
    </div>
  );
}
