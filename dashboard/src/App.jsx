import { useMemo, useState } from "react";
import {
  Download,
  FileText,
  Moon,
  Search,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { analyzeText, fallbackDashboard } from "./api.js";
import { CaseNote } from "./components/CaseNote.jsx";
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
    <main className={`app case-file-app ${theme}`}>
      <header className="topbar case-file-topbar">
        <div className="brand">
          <div>
            <p>C-SSRS severity research file</p>
            <h1>Case Evidence Review</h1>
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

      <section className="case-intro">
        <div>
          <span className="case-label">
            <ShieldCheck size={14} /> Evidence-linked output
          </span>
          <h2>Trace a post from highlighted wording to mapped concepts, evidence excerpts, and support pathways.</h2>
          <p>
            This screen is a research case review. It shows what the model detected and which
            graph-linked documents support the suggested pathway.
          </p>
        </div>
        <div className="file-tab">
          <span>Case opened</span>
          <strong>{new Date().toLocaleString()}</strong>
        </div>
      </section>

      <section className="case-input-row">
        <div className="text-entry">
          <label htmlFor="clinical-text">Edit submitted post</label>
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

      <section className="case-file-grid">
        <section className="case-file-main">
          <CaseNote
            text={dashboard.inputPreview || text}
            concepts={dashboard.concepts}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
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
        </section>

        <aside className="case-file-sidebar">
          <PredictionPanel prediction={dashboard.prediction} />
          <ShapExplorer
            explainability={dashboard.explainability}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
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
