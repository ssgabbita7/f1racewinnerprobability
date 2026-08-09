import StartLights from "./StartLights";

export default function Header() {
  return (
    <div className="mb-1">
      <div className="text-center">
        <div className="font-display text-[0.72rem] tracking-[0.4em] text-f1-cyan uppercase opacity-85 mb-1">
          2026 Season &middot; Telemetry-Driven
        </div>
        <h1 className="f1-title font-display font-black text-4xl sm:text-5xl tracking-wide leading-tight m-0">
          F1 WIN PROBABILITY
        </h1>
        <div className="w-36 h-1 mx-auto my-4 rounded-full bg-gradient-to-r from-transparent via-f1-red to-transparent" />
        <p className="text-zinc-400 text-sm max-w-md mx-auto leading-relaxed">
          Ask a natural-language question about a live race scenario and get a
          data-driven win probability backed by historical F1 results.
        </p>
      </div>
      <StartLights />
    </div>
  );
}
