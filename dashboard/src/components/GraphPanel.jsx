import ReactFlow, { Background, Controls } from "reactflow";
import { GitBranch, Route } from "lucide-react";

function nodeColor(type) {
  if (type === "SHAP Token") return "#38bdf8";
  if (type === "Symptom" || type === "Emotion") return "#22c55e";
  if (type === "Intervention") return "#f59e0b";
  if (type === "Evidence") return "#a855f7";
  return "#94a3b8";
}

function flowElements(graph) {
  const nodes = graph.nodes.map((node, index) => ({
    id: node.id,
    data: { label: `${node.label}\n${node.type}` },
    position: { x: (index % 2) * 230, y: Math.floor(index / 2) * 105 },
    style: {
      border: `1px solid ${nodeColor(node.type)}`,
      background: "rgba(15, 23, 42, 0.86)",
      color: "#f8fafc",
      borderRadius: 10,
      padding: 10,
      width: 190,
      fontSize: 12,
      whiteSpace: "pre-line",
    },
  }));
  const edges = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label,
    animated: true,
    style: { stroke: "#38bdf8" },
    labelStyle: { fill: "#cbd5e1", fontSize: 10 },
  }));
  return { nodes, edges };
}

export function GraphPanel({ graph, selectedToken }) {
  const { nodes, edges } = flowElements(graph);
  return (
    <section className="panel graph-panel">
      <div className="panel-heading">
        <span><GitBranch size={16} /> Knowledge Graph</span>
      </div>
      <div className="graph-canvas">
        <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
          <Background color="#334155" gap={18} />
          <Controls />
        </ReactFlow>
      </div>
      <div className="trace-box">
        <h3><Route size={15} /> Graph trace</h3>
        {graph.trace.map((step, index) => (
          <div className={`trace-step ${step.token === selectedToken ? "active" : ""}`} key={`${step.token}-${index}`}>
            <span>{step.token}</span>
            <span>{step.concept}</span>
            <span>{step.intervention}</span>
            <span>{step.evidence}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
