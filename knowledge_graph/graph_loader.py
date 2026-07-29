"""Neo4j connection and constraint utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password_env: str
    database: str

    @classmethod
    def from_config(cls, cfg: Any) -> "Neo4jSettings":
        return cls(
            uri=str(cfg.neo4j.uri),
            username=str(cfg.neo4j.username),
            password_env=str(cfg.neo4j.password_env),
            database=str(cfg.neo4j.database),
        )

    def password(self) -> str:
        value = os.environ.get(self.password_env)
        if not value:
            raise RuntimeError(
                f"Missing Neo4j password environment variable: {self.password_env}"
            )
        return value


def open_driver(settings: Neo4jSettings):
    """Open a Neo4j driver using settings loaded from project config."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password()),
    )


def create_constraints(driver, database: str) -> None:
    """Create idempotent Neo4j constraints for seeded graph entities."""
    statements = [
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (n:Emotion) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT intervention_name IF NOT EXISTS FOR (n:Intervention) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT evidence_name IF NOT EXISTS FOR (n:EvidenceSource) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT resource_name IF NOT EXISTS FOR (n:Resource) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT alias_key IF NOT EXISTS FOR (n:Alias) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT severity_level IF NOT EXISTS FOR (n:SeverityBand) REQUIRE n.level IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for statement in statements:
            session.run(statement)
