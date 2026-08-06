import { FileText, Highlighter } from "lucide-react";

function splitWithHighlights(text, tokens) {
  const phrases = tokens.filter(Boolean).sort((a, b) => b.length - a.length);
  if (!phrases.length) return [text];
  const escaped = phrases.map((phrase) => phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return text.split(pattern).filter(Boolean);
}

export function CaseNote({ text, concepts, selectedToken, onSelectToken }) {
  const tokens = concepts.map((concept) => concept.matched_alias);
  const parts = splitWithHighlights(text, tokens);

  return (
    <section className="case-note">
      <div className="case-note-head">
        <span><FileText size={16} /> Submitted post</span>
        <span><Highlighter size={15} /> Detected phrases are highlighted</span>
      </div>
      <div className="case-note-paper">
        {parts.map((part, index) => {
          const matchingConcept = concepts.find(
            (concept) => concept.matched_alias.toLowerCase() === part.toLowerCase(),
          );
          if (!matchingConcept) {
            return <span key={`${part}-${index}`}>{part}</span>;
          }
          return (
            <button
              className={selectedToken === matchingConcept.matched_alias ? "note-highlight active" : "note-highlight"}
              key={`${part}-${index}`}
              onClick={() => onSelectToken(matchingConcept.matched_alias)}
              title={`${matchingConcept.matched_alias} maps to ${matchingConcept.name}`}
            >
              {part}
            </button>
          );
        })}
      </div>
      <div className="case-concepts">
        {concepts.map((concept) => (
          <button
            key={`${concept.name}-${concept.matched_alias}`}
            className={selectedToken === concept.matched_alias ? "concept-note active" : "concept-note"}
            onClick={() => onSelectToken(concept.matched_alias)}
          >
            <span>{concept.matched_alias}</span>
            <strong>{concept.name}</strong>
          </button>
        ))}
      </div>
    </section>
  );
}
