export default function ContextGrid({ parsed = {} }) {
  const fields = [
    ["Driver", parsed.driver, "🏎"],
    ["Circuit", parsed.circuit, "🏁"],
    ["Weather", parsed.weather, "🌦"],
    ["Current position", parsed.current_position, "📍"],
    ["Lap", parsed.lap ? `${parsed.lap} / ${parsed.total_laps}` : null, "⏱"],
    [
      "Race progress",
      parsed.race_progress_pct ? `${Math.round(parsed.race_progress_pct * 100)}%` : null,
      "📊",
    ],
    ["Grid position", parsed.grid_position, "🚦"],
    ["Team", parsed.team, "🛠"],
    ["Year", parsed.year, "📅"],
  ].filter(([, value]) => value !== null && value !== undefined);

  return (
    <div>
      <div className="f1-section-title mb-3">Understood Race Context</div>
      <div className="grid gap-3 grid-cols-[repeat(auto-fit,minmax(140px,1fr))]">
        {fields.map(([label, value, icon]) => (
          <div className="f1-stat" key={label}>
            <div className="f1-stat-label">
              {icon} {label}
            </div>
            <div className="f1-stat-value">{value}</div>
          </div>
        ))}
      </div>
      {parsed.notes && <div className="text-zinc-400 text-sm mt-3">Notes: {parsed.notes}</div>}
    </div>
  );
}
