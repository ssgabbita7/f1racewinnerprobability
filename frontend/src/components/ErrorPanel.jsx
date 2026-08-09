export default function ErrorPanel({ error }) {
  return (
    <div className="mt-6 f1-panel px-4 py-4" style={{ borderColor: "rgba(255,30,30,0.4)" }}>
      <div className="font-bold text-f1-red mb-1">{error.message}</div>
      {error.detail && <div className="text-zinc-400 text-sm">{error.detail}</div>}
      {error.hint && (
        <div className="mt-2 text-sm text-f1-cyan bg-f1-cyan/10 border border-f1-cyan/30 rounded-lg px-3 py-2">
          {error.hint}
        </div>
      )}
    </div>
  );
}
