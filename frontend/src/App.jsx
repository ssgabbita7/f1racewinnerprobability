import { useState } from "react";
import Header from "./components/Header";
import ExampleQueries from "./components/ExampleQueries";
import QueryForm from "./components/QueryForm";
import ResultsPanel from "./components/ResultsPanel";
import ErrorPanel from "./components/ErrorPanel";
import { predict } from "./lib/api";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predict(query);
      setResult(data);
    } catch (err) {
      setError({ message: err.message, detail: err.detail, hint: err.hint });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen px-4 py-10">
      <div className="max-w-3xl mx-auto">
        <Header />
        <ExampleQueries onSelect={setQuery} />
        <QueryForm
          query={query}
          onQueryChange={setQuery}
          onSubmit={handleSubmit}
          loading={loading}
        />
        {error && <ErrorPanel error={error} />}
        {result && <ResultsPanel result={result} />}
      </div>
    </div>
  );
}

export default App;
