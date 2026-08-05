import { ArrowRight, BookOpenCheck, ClipboardList } from "lucide-react";

export function PathwayPanel({ pathways, selectedToken, onSelectToken }) {
  const rows = pathways.map((pathway) => ({
    detectedText: pathway.detectedText || pathway.token,
    mappedConcept: pathway.mappedConcept || pathway.concept,
    conceptType: pathway.conceptType || "Concept",
    guidance: pathway.guidance || pathway.intervention,
    evidenceSource: pathway.evidenceSource || pathway.evidence,
    whySelected:
      pathway.whySelected ||
      `${pathway.concept} is connected to ${pathway.intervention} through graph traversal.`,
    evidenceSnippet: pathway.evidenceSnippet || "",
    sourceUrl: pathway.sourceUrl || "",
  }));
  return (
    <section className="panel pathway-panel">
      <div className="panel-heading">
        <span><ClipboardList size={16} /> Decision Pathway</span>
      </div>
      <p className="muted-copy">
        Each pathway is produced from SHAP token mapping and Neo4j traversal.
      </p>
      <div className="pathway-list">
        {rows.map((pathway, index) => (
          <article
            className={`pathway-card ${pathway.detectedText === selectedToken ? "selected" : ""}`}
            key={`${pathway.detectedText}-${pathway.guidance}-${index}`}
          >
            <button className="pathway-token" onClick={() => onSelectToken(pathway.detectedText)}>
              {pathway.detectedText}
            </button>
            <div className="pathway-flow">
              <span>{pathway.mappedConcept}</span>
              <ArrowRight size={15} />
              <span>{pathway.guidance}</span>
              <ArrowRight size={15} />
              <span>{pathway.evidenceSource}</span>
            </div>
            <p>{pathway.whySelected}</p>
            {pathway.evidenceSnippet && (
              <blockquote>
                <BookOpenCheck size={15} />
                {pathway.evidenceSnippet}
              </blockquote>
            )}
            {pathway.sourceUrl && (
              <a href={pathway.sourceUrl} target="_blank" rel="noreferrer">
                Open source document
              </a>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
