const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

export async function predict(query) {
  let res;
  try {
    res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch {
    throw new Error(
      `Cannot connect to the backend at ${BACKEND_URL}. Is the Node.js server running?`
    );
  }

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const err = new Error(data.error || "Unknown error");
    err.detail = data.detail;
    err.hint = data.hint;
    throw err;
  }

  return data;
}
