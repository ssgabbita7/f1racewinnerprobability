/**
 * POST /predict
 *
 * Accepts: { query: string }
 * Returns: {
 *   probability: number,         // blended win probability 0–1
 *   probability_pct: string,     // e.g. "72%"
 *   confidence: string,          // "low" | "medium" | "high"
 *   parsed_context: object,      // extracted entities from NLP parsing
 *   supporting_cases: Array,     // 2–3 historical examples
 *   debug: {                     // raw signals for transparency
 *     knn_win_rate: number,
 *     model_probability: number,
 *   }
 * }
 */
const { Router } = require("express");
const { parseQuery, entitiesToFeatures, buildQueryText } = require("../services/nlpParser");
const mlService = require("../services/mlService");

const router = Router();

router.post("/predict", async (req, res, next) => {
  try {
    const { query } = req.body;

    if (!query || typeof query !== "string" || query.trim().length === 0) {
      return res.status(400).json({
        error: "Missing or empty 'query' field. Provide a natural-language race query.",
      });
    }

    // ── 1. Check ML service availability ─────────────────────────────────────
    const ready = await mlService.isReady();
    if (!ready) {
      return res.status(503).json({
        error:
          "The ML service is not ready. " +
          "Make sure the Python microservice is running and the data pipeline has been executed.",
      });
    }

    // ── 2. NLP: extract structured entities from the query ───────────────────
    let entities;
    try {
      entities = await parseQuery(query);
    } catch (nlpErr) {
      return res.status(422).json({
        error: "Could not parse your query.",
        detail: nlpErr.message,
        hint: "Try phrasing like: 'Hamilton is leading at Silverstone in the wet on lap 40 of 52'",
      });
    }

    // ── 3. Check we extracted at minimum a driver or a circuit ───────────────
    if (!entities.driver && !entities.circuit) {
      return res.status(422).json({
        error: "Could not identify a driver or circuit in your query.",
        parsed_context: entities,
        hint: "Please mention at least a driver name or race location.",
      });
    }

    // ── 4. Build query text and features for the ML service ──────────────────
    const queryText = buildQueryText(entities);
    const features = entitiesToFeatures(entities);

    // ── 5. Call Python ML service ─────────────────────────────────────────────
    let mlResult;
    try {
      mlResult = await mlService.predict(queryText, features);
    } catch (mlErr) {
      return res.status(502).json({
        error: "ML service prediction failed.",
        detail: mlErr.message,
      });
    }

    // ── 6. Compose response ───────────────────────────────────────────────────
    const probability = mlResult.blended_probability;
    const pct = Math.round(probability * 100);

    return res.json({
      probability,
      probability_pct: `${pct}%`,
      confidence: mlResult.confidence,
      parsed_context: entities,
      supporting_cases: mlResult.supporting_cases,
      debug: {
        knn_win_rate: mlResult.knn_win_rate,
        model_probability: mlResult.model_probability,
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
