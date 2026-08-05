import { ArrowDown, ArrowUp, MousePointerClick } from "lucide-react";

function TokenButton({ factor, selected, onSelect }) {
  const isPositive = factor.direction === "positive";
  return (
    <button
      className={`token-button ${isPositive ? "positive" : "negative"} ${selected ? "selected" : ""}`}
      onClick={() => onSelect(factor.token)}
      title={`${factor.token}: ${factor.value.toFixed(4)}`}
    >
      {isPositive ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
      <span>{factor.token}</span>
      <strong>{factor.value.toFixed(4)}</strong>
    </button>
  );
}

export function ShapExplorer({ explainability, selectedToken, onSelectToken }) {
  return (
    <section className="panel shap-panel">
      <div className="panel-heading">
        <span><MousePointerClick size={16} /> SHAP Explainability</span>
      </div>
      <p className="muted-copy">Click a token to filter concepts, evidence, and graph trace.</p>
      <div className="token-group">
        <h3>Positive contribution</h3>
        {explainability.positiveTokens.map((factor) => (
          <TokenButton
            key={`${factor.token}-${factor.value}`}
            factor={factor}
            selected={selectedToken === factor.token}
            onSelect={onSelectToken}
          />
        ))}
      </div>
      <div className="token-group">
        <h3>Negative contribution</h3>
        {explainability.negativeTokens.map((factor) => (
          <TokenButton
            key={`${factor.token}-${factor.value}`}
            factor={factor}
            selected={selectedToken === factor.token}
            onSelect={onSelectToken}
          />
        ))}
      </div>
    </section>
  );
}

