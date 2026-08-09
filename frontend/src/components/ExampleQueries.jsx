import { useState } from "react";

const EXAMPLES = [
  "It just rained at Silverstone, Hamilton is in the lead, what is the probability of his win?",
  "Verstappen started on pole at Monza, lap 30 of 53, dry conditions, currently P1",
  "Alonso is P3 at Monaco on lap 55 of 78 in mixed weather, what are his chances?",
];

export default function ExampleQueries({ onSelect }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="f1-panel mb-4 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left font-display text-xs tracking-wide uppercase text-zinc-200"
      >
        <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>
          &rsaquo;
        </span>
        🏁 Example queries
      </button>
      {open && (
        <div className="px-4 pb-4 flex flex-col gap-2">
          {EXAMPLES.map((ex) => (
            <button
              type="button"
              key={ex}
              onClick={() => onSelect(ex)}
              className="text-left text-sm px-3 py-2 rounded-lg bg-white/4 border border-white/8 text-zinc-200 hover:border-f1-cyan hover:text-white transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
