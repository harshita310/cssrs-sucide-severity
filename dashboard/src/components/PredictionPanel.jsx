import { BadgeCheck, BrainCircuit } from "lucide-react";

export function PredictionPanel({ prediction }) {
  const topProbabilities = prediction.probabilities
    .map((value, index) => ({ label: index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  return (
    <section className="panel prediction-panel assessment-stamp">
      <div className="panel-heading">
        <span><BrainCircuit size={16} /> Assessment stamp</span>
        <BadgeCheck size={18} />
      </div>
      <div className={`risk-card risk-${prediction.riskLevel.toLowerCase().split(" ")[0]}`}>
        <p>Predicted severity band</p>
        <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
        <span>{prediction.riskLevel} | label {prediction.severityLabel}</span>
      </div>
      <div className="model-card">
        <p>Checkpoint</p>
        <strong>{prediction.modelVersion}</strong>
        <span>{prediction.validatedModel ? "Validated checkpoint selected" : "Validation status unavailable"}</span>
      </div>
      <div className="probability-table">
        {topProbabilities.map((item) => (
          <div key={item.label}>
            <span>Label {item.label}</span>
            <meter value={item.value} min="0" max="1" />
            <strong>{(item.value * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
