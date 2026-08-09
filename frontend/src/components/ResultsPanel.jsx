import Gauge from "./Gauge";
import ConfidenceBadge from "./ConfidenceBadge";
import ContextGrid from "./ContextGrid";
import CaseCard from "./CaseCard";

export default function ResultsPanel({ result }) {
  const {
    probability,
    probability_pct,
    confidence,
    parsed_context = {},
    supporting_cases = [],
    debug = {},
  } = result;

  return (
    <div className="mt-8 flex flex-col gap-8">
      <div className="grid grid-cols-1 sm:grid-cols-[2fr_1fr] gap-6 items-start pt-6 border-t border-white/8">
        <Gauge prob={probability} pct={probability_pct} debug={debug} />
        <ConfidenceBadge confidence={confidence} />
      </div>

      <div className="pt-6 border-t border-white/8">
        <ContextGrid parsed={parsed_context} />
      </div>

      {supporting_cases.length > 0 && (
        <div className="pt-6 border-t border-white/8">
          <div className="f1-section-title mb-3">Supporting Historical Cases</div>
          {supporting_cases.map((c, i) => (
            <CaseCard key={`${c.year}-${c.driver_name}-${i}`} case={c} />
          ))}
        </div>
      )}
    </div>
  );
}
