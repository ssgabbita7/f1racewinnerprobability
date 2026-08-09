require("dotenv").config({ path: require("path").join(__dirname, "../../.env") });
const express = require("express");
const cors = require("cors");

const predictRouter = require("./routes/predict");

const app = express();
const PORT = process.env.NODE_PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "f1-predictor-backend" });
});

app.use("/", predictRouter);

app.use((err, _req, res, _next) => {
  console.error("[error]", err.message);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`F1 Predictor backend listening on http://localhost:${PORT}`);
});
