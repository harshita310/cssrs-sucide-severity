import { ExternalLink, FileSearch, Highlighter } from "lucide-react";

function highlightSnippet(snippet, selectedToken, selectedConcept) {
  const terms = [selectedToken, selectedConcept?.name, selectedConcept?.matched_alias]
    .filter(Boolean)
    .map((term) => term.toLowerCase());
  const words = snippet.split(/(\s+)/);
  return words.map((word, index) => {
    const clean = word.toLowerCase().replace(/[^a-z0-9]/g, "");
    const matched = terms.some((term) => term.split(/\s+/).some((part) => part && clean.includes(part)));
    return matched ? <mark key={`${word}-${index}`}>{word}</mark> : word;
  });
}

export function EvidenceExplorer({ evidence, selectedToken, selectedConcept }) {
  return (
    <section className="workspace-panel">
      <div className="workspace-heading">
        <div>
          <p>Evidence Explorer</p>
          <h2>Retrieved published evidence</h2>
        </div>
        <span className="context-pill">
          <Highlighter size={14} />
          {selectedConcept?.name || selectedToken || "All evidence"}
        </span>
      </div>
      <div className="evidence-list">
        {evidence.map((item) => (
          <article className="evidence-card" key={item.id}>
            <div className="evidence-title">
              <FileSearch size={18} />
              <div>
                <h3>{item.title}</h3>
                <p>{item.organization} | {item.evidenceLevel} | {item.publicationYear}</p>
              </div>
            </div>
            <div className="metric-row">
              <span>Similarity {Number(item.similarityScore).toFixed(2)}</span>
              <span>Confidence {Number(item.confidence).toFixed(2)}</span>
              <span>{item.section}</span>
            </div>
            <p className="snippet">{highlightSnippet(item.snippet, selectedToken, selectedConcept)}</p>
            <div className="evidence-footer">
              <span>Supports {item.supports}</span>
              {item.sourceUrl && (
                <a href={item.sourceUrl} target="_blank" rel="noreferrer">
                  Open source <ExternalLink size={14} />
                </a>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

