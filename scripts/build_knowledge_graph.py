"""Create Neo4j constraints and load the XAI clinical seed graph."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.graph_builder import load_seed_graph
from knowledge_graph.graph_loader import Neo4jSettings, create_constraints, open_driver
from utils import load_config


def main() -> None:
    cfg = load_config()
    settings = Neo4jSettings.from_config(cfg)
    driver = open_driver(settings)
    try:
        create_constraints(driver, settings.database)
        count = load_seed_graph(driver, settings.database)
    finally:
        driver.close()

    print(
        "Neo4j knowledge graph loaded | "
        f"database={settings.database} statements={count}"
    )


if __name__ == "__main__":
    main()
