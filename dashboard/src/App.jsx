import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Download,
  FileText,
  Moon,
  Search,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { analyzeText, fallbackDashboard } from "./api.js";
import { EvidenceExplorer } from "./components/EvidenceExplorer.jsx";
import { LiteraturePanel } from "./components/LiteraturePanel.jsx";
import { PathwayPanel } from "./components/PathwayPanel.jsx";
import { PredictionPanel } from "./components/PredictionPanel.jsx";
import { RecommendationExplorer } from "./components/RecommendationExplorer.jsx";
import { ShapExplorer } from "./components/ShapExplorer.jsx";

const DEFAULT_TEXT = "I cannot sleep and I feel alone";

export default function App() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [dashboard, setDashboard] = useState(fallbackDashboard());
  const [selectedToken, setSelectedToken] = useState("cannot sleep");
  const [theme, setTheme] = useState("light");
  const [isLoading, setIsLoading] = useState(false);
  const [apiNotice, setApiNotice] = useState("Preview mode: sample payload loaded.");
  const [tab, setTab] = useState("evidence");

  const selectedConcept = useMemo(
    () =>
      dashboard.concepts.find(
        (concept) => concept.matched_alias === selectedToken || concept.name === selectedToken,
      ),
    [dashboard.concepts, selectedToken],
  );

  async function handleAnalyze() {
    setIsLoading(true);
    setApiNotice("Running model prediction, token attribution, graph lookup, and evidence retrieval...");
    try {
      const result = await analyzeText(text);
      setDashboard(result);
      const firstToken =
        result.explainability.positiveTokens[0]?.token ||
        result.explainability.negativeTokens[0]?.token ||
        "";
      setSelectedToken(firstToken);
      setApiNotice("Live result loaded from model, graph, and retrieved evidence.");
    } catch (error) {
      setDashboard(fallbackDashboard());
      setApiNotice(`API unavailable, showing sample payload. ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  function openHtmlReport() {
    const path = dashboard.exports?.htmlReport;
    if (!path) {
      window.print();
      return;
    }
    const encoded = encodeURIComponent(path);
    window.open(`http://127.0.0.1:8000/api/report/html?path=${encoded}`, "_blank");
  }

  return (
    <main className={`app ${theme}`}>
      <div className="orbital-grid" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={22} />
          </div>
          <div>
          <p>CSSRS research system</p>
          <h1>Clinical Evidence Review</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="ghost-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <button className="ghost-button" onClick={openHtmlReport}>
            <FileText size={16} />
            Report
          </button>
          <button className="primary-button" onClick={() => window.print()}>
            <Download size={16} />
            PDF
          </button>
        </div>
      </header>

      <section className="hero-strip">
        <div>
          <span className="eyebrow">
            <ShieldCheck size={14} /> Evidence-linked output
          </span>
          <h2>Review detected language, mapped clinical concepts, and source-backed support pathways.</h2>
        </div>
        <div className="patient-chip">
          <span>Research Case</span>
          <strong>{new Date().toLocaleString()}</strong>
        </div>
      </section>

      <section className="analysis-console">
        <div className="text-entry">
          <label htmlFor="clinical-text">Input text</label>
          <textarea
            id="clinical-text"
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={3}
          />
        </div>
        <button className="analyze-button" onClick={handleAnalyze} disabled={isLoading}>
          <Search size={18} />
          {isLoading ? "Analyzing" : "Analyze"}
        </button>
      </section>
      <p className="api-notice">{apiNotice}</p>

      <section className="dashboard-grid">
        <aside className="left-rail">
          <PredictionPanel prediction={dashboard.prediction} />
          <ShapExplorer
            explainability={dashboard.explainability}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
        </aside>

        <motion.section
          className="center-stage"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="segmented-tabs">
            {["evidence", "recommendations", "literature"].map((item) => (
              <button
                key={item}
                className={tab === item ? "active" : ""}
                onClick={() => setTab(item)}
              >
                {item}
              </button>
            ))}
          </div>
          {tab === "evidence" && (
            <EvidenceExplorer
              evidence={dashboard.evidence}
              selectedToken={selectedToken}
              selectedConcept={selectedConcept}
            />
          )}
          {tab === "recommendations" && (
            <RecommendationExplorer recommendations={dashboard.recommendations} />
          )}
          {tab === "literature" && <LiteraturePanel literature={dashboard.literature} />}
        </motion.section>

        <aside className="right-rail">
          <PathwayPanel
            pathways={dashboard.pathways || dashboard.graph.trace}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
        </aside>
      </section>
    </main>
  );
}
