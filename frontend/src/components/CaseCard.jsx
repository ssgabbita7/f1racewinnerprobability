export default function CaseCard({ case: c }) {
  const finished = c.finished_position;

  let outcome, accent;
  if (c.won) {
    outcome = "🏆 WIN";
    accent = "var(--color-f1-green)";
  } else if (!finished) {
    outcome = "💀 DNF";
    accent = "var(--color-f1-red)";
  } else {
    outcome = `P${finished}`;
    accent = "var(--color-f1-amber)";
  }

  const race = c.race_name || c.circuit_name || "?";
  const gridStr = c.grid_position ? ` · Grid P${c.grid_position}` : "";
  const weather = String(c.weather ?? "?");
  const weatherLabel = weather.charAt(0).toUpperCase() + weather.slice(1);

  return (
    <div className="f1-case mb-3" style={{ "--case-accent": accent }}>
      <div className="flex-1">
        <div className="font-bold text-sm text-zinc-100 mb-0.5">
          {c.year ?? "?"} {race} — {c.driver_name ?? "?"}
          {gridStr} · {weatherLabel} conditions
        </div>
        <div className="text-zinc-400 text-sm">{c.scenario_text}</div>
      </div>
      <div className="font-display text-lg font-extrabold whitespace-nowrap" style={{ color: accent }}>
        {outcome}
      </div>
    </div>
  );
}
