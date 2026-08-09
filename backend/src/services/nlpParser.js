/**
 * NLP entity extraction via Claude API (tool-use / function-calling).
 *
 * Input : raw natural-language query string
 * Output: structured JSON matching ENTITY_SCHEMA
 */
const Anthropic = require("@anthropic-ai/sdk");

const client = new Anthropic.default({ apiKey: process.env.ANTHROPIC_API_KEY });
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || "claude-sonnet-4-6";

/** The strict JSON schema Claude must fill in. */
const ENTITY_SCHEMA = {
  type: "object",
  properties: {
    driver: {
      type: ["string", "null"],
      description: "Canonical driver surname or full name, e.g. 'hamilton', 'verstappen'",
    },
    circuit: {
      type: ["string", "null"],
      description: "Circuit or location slug, e.g. 'silverstone', 'monza', 'monaco'",
    },
    weather: {
      type: ["string", "null"],
      enum: ["dry", "wet", "mixed", null],
      description: "Weather condition: 'wet', 'dry', 'mixed', or null if not mentioned",
    },
    current_position: {
      type: ["integer", "null"],
      description: "Driver's current race position, e.g. 1 if leading",
    },
    lap: {
      type: ["integer", "null"],
      description: "Current lap number mentioned in the query",
    },
    total_laps: {
      type: ["integer", "null"],
      description: "Total laps in the race if mentioned",
    },
    race_progress_pct: {
      type: ["number", "null"],
      description: "Fraction of race completed (0–1), derived from lap/total_laps if available",
    },
    grid_position: {
      type: ["integer", "null"],
      description: "Driver's qualifying/starting grid position if mentioned",
    },
    team: {
      type: ["string", "null"],
      description: "Constructor/team name if mentioned, e.g. 'mercedes', 'red bull'",
    },
    year: {
      type: ["integer", "null"],
      description: "Season year if mentioned",
    },
    notes: {
      type: ["string", "null"],
      description: "Any ambiguous context extracted verbatim that could affect the prediction",
    },
    safety_car: {
      type: ["boolean", "null"],
      description: "true if the query mentions a full (not virtual) safety car is/was deployed",
    },
    virtual_safety_car: {
      type: ["boolean", "null"],
      description: "true if the query mentions a virtual safety car (VSC) is/was active",
    },
    pit_stops_completed: {
      type: ["integer", "null"],
      description: "Number of pit stops the driver has completed so far, if stated or impliable (e.g. 'after his second stop' → 2)",
    },
  },
  required: [
    "driver",
    "circuit",
    "weather",
    "current_position",
    "lap",
    "total_laps",
    "race_progress_pct",
    "grid_position",
    "team",
    "year",
    "notes",
    "safety_car",
    "virtual_safety_car",
    "pit_stops_completed",
  ],
};

const EXTRACT_TOOL = {
  name: "extract_race_context",
  description:
    "Extract structured F1 race context from a natural-language query. " +
    "Return null for any field not mentioned or inferable from the query.",
  input_schema: ENTITY_SCHEMA,
};

const SYSTEM_PROMPT = `You are an F1 race-context extractor.
Given a user query about a Formula 1 race scenario, extract the structured entities defined in the extract_race_context tool.
Rules:
- Normalise driver names to lowercase surname (e.g. "Lewis Hamilton" → "hamilton").
- Normalise circuit names to lowercase slug (e.g. "Silverstone" → "silverstone").
- If lap and total_laps are both present, compute race_progress_pct = lap / total_laps.
- If the user says "leading" or "in the lead" → current_position = 1.
- If the user mentions a full safety car (not virtual/VSC) → safety_car = true.
- If the user mentions "VSC" or "virtual safety car" → virtual_safety_car = true.
- If the user states a specific pit-stop count or ordinal stop (e.g. "after his second stop", "made 3 stops") → pit_stops_completed = that count.
- Use null for any field that cannot be determined from the query.
- Do NOT guess or hallucinate values not in the query.`;

/**
 * @param {string} query
 * @returns {Promise<object>} extracted entities
 */
async function parseQuery(query) {
  const response = await client.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    tools: [EXTRACT_TOOL],
    tool_choice: { type: "tool", name: "extract_race_context" },
    messages: [{ role: "user", content: query }],
  });

  const toolUse = response.content.find((b) => b.type === "tool_use");
  if (!toolUse) {
    throw new Error("Claude did not return a tool_use block — entity extraction failed.");
  }

  return toolUse.input;
}

/**
 * Convert parsed entities to the feature dict expected by the Python ML service.
 */
function entitiesToFeatures(entities) {
  const features = {};

  if (entities.grid_position != null) {
    features.grid_position = entities.grid_position;
  }
  if (entities.current_position != null) {
    features.current_position = entities.current_position;
  }
  if (entities.weather != null) {
    features.weather = entities.weather;
  }
  if (entities.race_progress_pct != null) {
    features.race_progress_pct = entities.race_progress_pct;
  } else if (entities.lap != null && entities.total_laps != null) {
    features.race_progress_pct = entities.lap / entities.total_laps;
  }
  if (entities.safety_car != null) {
    features.safety_car_active = entities.safety_car;
  }
  if (entities.virtual_safety_car != null) {
    features.vsc_active = entities.virtual_safety_car;
  }
  if (entities.pit_stops_completed != null) {
    features.pit_stops_completed = entities.pit_stops_completed;
  }

  return features;
}

/**
 * Build the scenario text sent to the Python service for FAISS embedding.
 * Mirrors the format used in build_scenarios.py so embeddings are comparable.
 */
function buildQueryText(entities) {
  const parts = [];
  if (entities.year) parts.push(String(entities.year));
  if (entities.circuit) parts.push(`race at ${entities.circuit}`);
  if (entities.driver) parts.push(entities.driver);
  if (entities.team) parts.push(`driving for ${entities.team}`);
  if (entities.grid_position) parts.push(`started P${entities.grid_position}`);
  if (entities.current_position) parts.push(`currently P${entities.current_position}`);
  if (entities.weather) parts.push(`${entities.weather} conditions`);
  if (entities.lap && entities.total_laps) {
    parts.push(`lap ${entities.lap} of ${entities.total_laps}`);
  } else if (entities.lap) {
    parts.push(`lap ${entities.lap}`);
  }
  if (entities.safety_car) parts.push("safety car deployed");
  if (entities.virtual_safety_car) parts.push("virtual safety car active");
  if (entities.pit_stops_completed != null) {
    parts.push(`completed ${entities.pit_stops_completed} pit stops`);
  }
  return parts.join(", ");
}

module.exports = { parseQuery, entitiesToFeatures, buildQueryText };
