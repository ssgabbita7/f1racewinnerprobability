const STYLES = {
  low: { color: "var(--color-f1-red)", bg: "rgba(255,30,30,0.12)", icon: "🔴" },
  medium: { color: "var(--color-f1-amber)", bg: "rgba(255,201,60,0.12)", icon: "🟡" },
  high: { color: "var(--color-f1-green)", bg: "rgba(57,255,106,0.12)", icon: "🟢" },
};
const DEFAULT_STYLE = { color: "#9195AC", bg: "rgba(145,149,172,0.12)", icon: "⚪" };

export default function ConfidenceBadge({ confidence }) {
  const s = STYLES[confidence] ?? DEFAULT_STYLE;
  const label = confidence ? confidence.charAt(0).toUpperCase() + confidence.slice(1) : "Unknown";

  return (
    <div>
      <div className="font-display text-xs tracking-[0.25em] uppercase text-zinc-400 mb-1">
        Confidence
      </div>
      <div
        className="inline-flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wide px-4 py-2 rounded-lg border"
        style={{ color: s.color, background: s.bg, borderColor: `${s.color}55` }}
      >
        {s.icon} {label}
      </div>
    </div>
  );
}
