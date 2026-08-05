import { useMemo, useState } from "react";
import { Search } from "lucide-react";

const FILTERS = ["All", "WHO", "APA", "NICE", "CDC", "SAMHSA", "PubMed", "Nature", "Lancet", "BMJ"];

export function LiteraturePanel({ literature }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const filtered = useMemo(
    () =>
      literature.filter((item) => {
        const text = `${item.title} ${item.organization} ${item.snippet} ${item.sourceType}`.toLowerCase();
        const matchesQuery = text.includes(query.toLowerCase());
        const matchesFilter = filter === "All" || text.includes(filter.toLowerCase());
        return matchesQuery && matchesFilter;
      }),
    [literature, query, filter],
  );

  return (
    <section className="workspace-panel">
      <div className="workspace-heading">
        <div>
          <p>Clinical Literature</p>
          <h2>Search retrieved source chunks</h2>
        </div>
      </div>
      <div className="literature-controls">
        <label>
          <Search size={15} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title, source, snippet" />
        </label>
        <div className="filter-row">
          {FILTERS.map((item) => (
            <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>
              {item}
            </button>
          ))}
        </div>
      </div>
      <div className="literature-list">
        {filtered.map((item) => (
          <article key={item.id}>
            <strong>{item.title}</strong>
            <span>{item.sourceType} | similarity {Number(item.similarityScore).toFixed(2)}</span>
            <p>{item.snippet}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

