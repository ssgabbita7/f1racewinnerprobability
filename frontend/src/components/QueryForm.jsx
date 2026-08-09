export default function QueryForm({ query, onQueryChange, onSubmit, loading }) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="flex flex-col gap-3"
    >
      <label className="text-sm font-semibold text-zinc-300" htmlFor="query">
        Describe the race situation:
      </label>
      <textarea
        id="query"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="It just rained at Silverstone, Hamilton is in the lead, what is the probability of his win?"
        rows={4}
        className="w-full rounded-xl border border-white/12 bg-[#0f1016]/85 px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-f1-red focus:ring-2 focus:ring-f1-red/20 resize-none"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="font-display uppercase tracking-[0.12em] text-sm font-bold py-3 rounded-lg text-white bg-gradient-to-r from-f1-red-dim to-f1-red shadow-[0_4px_20px_rgba(255,30,30,0.35)] hover:-translate-y-px hover:shadow-[0_6px_26px_rgba(255,30,30,0.55)] transition-all disabled:opacity-50 disabled:pointer-events-none disabled:translate-y-0"
      >
        {loading ? "Analysing race context…" : "🏎 Predict"}
      </button>
    </form>
  );
}
