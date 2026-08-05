import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { BadgeCheck, BrainCircuit } from "lucide-react";

const COLORS = ["#38bdf8", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#f97316", "#14b8a6"];

export function PredictionPanel({ prediction }) {
  const chartData = prediction.probabilities.map((value, index) => ({
    name: `Label ${index}`,
    value: Number(value.toFixed(4)),
  }));

  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <span><BrainCircuit size={16} /> Clinical Prediction</span>
        <BadgeCheck size={18} />
      </div>
      <div className={`risk-card risk-${prediction.riskLevel.toLowerCase().split(" ")[0]}`}>
        <p>{prediction.riskLevel}</p>
        <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
        <span>Severity label {prediction.severityLabel}</span>
      </div>
      <div className="model-card">
        <p>Model Version</p>
        <strong>{prediction.modelVersion}</strong>
        <span>{prediction.validatedModel ? "Validated checkpoint selected" : "Validation status unavailable"}</span>
      </div>
      <div className="donut-wrap">
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={76} paddingAngle={2}>
              {chartData.map((entry, index) => (
                <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

