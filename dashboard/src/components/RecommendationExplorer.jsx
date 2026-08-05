import { CheckCircle2, Link2, MessageCircle, Phone, Video } from "lucide-react";

const supportIcons = [MessageCircle, Phone, Video];

export function RecommendationExplorer({ recommendations }) {
  return (
    <section className="workspace-panel">
      <div className="workspace-heading">
        <div>
          <p>Recommendation Explorer</p>
          <h2>Graph-selected support routes</h2>
        </div>
        <span className="context-pill">
          <Link2 size={14} />
          Neo4j traversal only
        </span>
      </div>
      <div className="recommendation-list">
        {recommendations.map((recommendation) => (
          <article className="recommendation-card" key={recommendation.name}>
            <div className="recommendation-head">
              <div>
                <h3>{recommendation.name}</h3>
                <p>{recommendation.purpose}</p>
              </div>
              <strong>{Number(recommendation.score).toFixed(2)}</strong>
            </div>
            <div className="linked-row">
              {recommendation.mappedConcepts.map((concept) => (
                <span key={concept}>{concept}</span>
              ))}
              {recommendation.supportingEvidence.map((evidence) => (
                <span key={evidence}>{evidence}</span>
              ))}
            </div>
            <div className="two-column">
              <div>
                <h4>Action steps</h4>
                {recommendation.actionSteps.map((step) => (
                  <p className="check-line" key={step}><CheckCircle2 size={15} /> {step}</p>
                ))}
              </div>
              <div>
                <h4>Support options</h4>
                {recommendation.supportOptions.map((option, index) => {
                  const Icon = supportIcons[index % supportIcons.length];
                  return <p className="check-line" key={option}><Icon size={15} /> {option}</p>;
                })}
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

