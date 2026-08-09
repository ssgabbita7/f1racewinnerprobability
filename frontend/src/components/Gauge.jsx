function colorFor(prob) {
  if (prob >= 0.6) return { color: "var(--color-f1-green)", glow: "rgba(57,255,106,0.45)" };
  if (prob >= 0.35) return { color: "var(--color-f1-amber)", glow: "rgba(255,201,60,0.45)" };
  return { color: "var(--color-f1-red)", glow: "rgba(255,30,30,0.45)" };
}

export default function Gauge({ prob, pct, debug = {} }) {
  const { color, glow } = colorFor(prob);

  return (
    <div>
      <div className="font-display text-xs tracking-[0.25em] uppercase text-zinc-400 mb-1">
        Win Probability
      </div>
      <div
        className="font-display font-black text-5xl mb-3 leading-none"
        style={{ color, textShadow: `0 0 24px ${glow}` }}
      >
        {pct}
      </div>
      <div className="f1-track">
        <div
          className="f1-fill"
          style={{
            width: `${Math.round(prob * 100)}%`,
            background: color,
            boxShadow: `0 0 16px ${glow}`,
          }}
        />
      </div>
      <div className="flex gap-2 flex-wrap mt-3">
        <span className="f1-chip">kNN win rate: {((debug.knn_win_rate ?? 0) * 100).toFixed(1)}%</span>
        <span className="f1-chip">Model: {((debug.model_probability ?? 0) * 100).toFixed(1)}%</span>
        <span className="f1-chip">Blended: {(prob * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
