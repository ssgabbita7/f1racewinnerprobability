const DELAYS = [0, 0.25, 0.5, 0.75, 1.0];

export default function StartLights() {
  return (
    <div className="flex justify-center gap-2.5 mb-6">
      {DELAYS.map((delay, i) => (
        <span
          key={i}
          className="w-3.5 h-3.5 rounded-full border border-white/10 animate-lightseq"
          style={{ animationDelay: `${delay}s` }}
        />
      ))}
    </div>
  );
}
