/**
 * HTTP client for the Python FastAPI ML microservice.
 */
const axios = require("axios");

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

/**
 * Call the Python /predict endpoint.
 *
 * @param {string} queryText   - scenario text for FAISS embedding
 * @param {object} features    - structured features for the sklearn model
 * @returns {Promise<{
 *   knn_win_rate: number,
 *   model_probability: number,
 *   blended_probability: number,
 *   confidence: string,
 *   supporting_cases: Array
 * }>}
 */
async function predict(queryText, features = {}) {
  try {
    const { data } = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      { query_text: queryText, features },
      { timeout: 30_000 }
    );
    return data;
  } catch (err) {
    if (err.response) {
      const detail = err.response.data?.detail || JSON.stringify(err.response.data);
      throw new Error(`ML service error (${err.response.status}): ${detail}`);
    }
    throw new Error(`ML service unreachable at ${ML_SERVICE_URL}: ${err.message}`);
  }
}

/**
 * Health-check the Python service.
 * @returns {Promise<boolean>}
 */
async function isReady() {
  try {
    const { data } = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 5_000 });
    return data.status === "ready";
  } catch {
    return false;
  }
}

module.exports = { predict, isReady };
