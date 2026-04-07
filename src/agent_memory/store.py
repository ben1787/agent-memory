from __future__ import annotations

from pathlib import Path

import kuzu

from agent_memory.models import MemoryRecord, SimilarityEdge


class GraphStore:
    def __init__(self, db_path: Path, dimensions: int, *, read_only: bool = False) -> None:
        self.db_path = db_path
        self.dimensions = dimensions
        self.read_only = read_only
        fresh = not db_path.exists()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path), read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        if fresh and not read_only:
            self._create_schema()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
        if getattr(self, "_db", None) is not None:
            self._db.close()
            self._db = None

    def _create_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE NODE TABLE Memory(
                id STRING PRIMARY KEY,
                text STRING,
                created_at STRING,
                importance DOUBLE,
                access_count INT64,
                last_accessed STRING,
                embedding FLOAT[{self.dimensions}]
            )
            """
        )
        self._conn.execute(
            """
            CREATE REL TABLE SIMILAR(
                FROM Memory TO Memory,
                weight DOUBLE,
                updated_at STRING
            )
            """
        )
        self._conn.execute(
            """
            CREATE REL TABLE NEXT(
                FROM Memory TO Memory,
                weight DOUBLE,
                updated_at STRING
            )
            """
        )

    def add_memory(self, memory: MemoryRecord) -> None:
        self._conn.execute(
            """
            CREATE (:Memory {
                id: $id,
                text: $text,
                created_at: $created_at,
                importance: $importance,
                access_count: $access_count,
                last_accessed: $last_accessed,
                embedding: $embedding
            })
            """,
            {
                "id": memory.id,
                "text": memory.text,
                "created_at": memory.created_at,
                "importance": memory.importance,
                "access_count": memory.access_count,
                "last_accessed": memory.last_accessed,
                "embedding": memory.embedding,
            },
        )

    def update_memory(self, memory: MemoryRecord) -> None:
        self._conn.execute(
            """
            MATCH (m:Memory {id: $id})
            SET m.text = $text,
                m.created_at = $created_at,
                m.importance = $importance,
                m.access_count = $access_count,
                m.last_accessed = $last_accessed,
                m.embedding = $embedding
            """,
            {
                "id": memory.id,
                "text": memory.text,
                "created_at": memory.created_at,
                "importance": memory.importance,
                "access_count": memory.access_count,
                "last_accessed": memory.last_accessed,
                "embedding": memory.embedding,
            },
        )

    def delete_memory(self, memory_id: str) -> None:
        # DETACH DELETE drops the node and all incident SIMILAR/NEXT relationships
        # in one statement; a plain DELETE would fail if any edges exist.
        self._conn.execute(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m",
            {"id": memory_id},
        )

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        rows = self._conn.execute(
            """
            MATCH (m:Memory {id: $id})
            RETURN
                m.id,
                m.text,
                m.created_at,
                m.importance,
                m.access_count,
                m.last_accessed,
                m.embedding
            LIMIT 1
            """,
            {"id": memory_id},
        ).get_all()
        if not rows:
            return None
        row = rows[0]
        return MemoryRecord(
            id=row[0],
            text=row[1],
            created_at=row[2],
            importance=float(row[3]),
            access_count=int(row[4]),
            last_accessed=row[5],
            embedding=[float(value) for value in row[6]],
        )

    def delete_similarity_edges_for(self, memory_id: str) -> None:
        # Drop both directions of SIMILAR edges incident on this memory.
        # Used when re-embedding (edit) so we can recompute fresh edges.
        self._conn.execute(
            "MATCH (m:Memory {id: $id})-[r:SIMILAR]->() DELETE r",
            {"id": memory_id},
        )
        self._conn.execute(
            "MATCH ()-[r:SIMILAR]->(m:Memory {id: $id}) DELETE r",
            {"id": memory_id},
        )

    def list_memories(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            """
            MATCH (m:Memory)
            RETURN
                m.id,
                m.text,
                m.created_at,
                m.importance,
                m.access_count,
                m.last_accessed,
                m.embedding
            ORDER BY m.created_at ASC
            """
        ).get_all()
        memories = []
        for row in rows:
            memories.append(
                MemoryRecord(
                    id=row[0],
                    text=row[1],
                    created_at=row[2],
                    importance=float(row[3]),
                    access_count=int(row[4]),
                    last_accessed=row[5],
                    embedding=[float(value) for value in row[6]],
                )
            )
        return memories

    def get_last_memory(self) -> MemoryRecord | None:
        rows = self._conn.execute(
            """
            MATCH (m:Memory)
            RETURN
                m.id,
                m.text,
                m.created_at,
                m.importance,
                m.access_count,
                m.last_accessed,
                m.embedding
            ORDER BY m.created_at DESC
            LIMIT 1
            """
        ).get_all()
        if not rows:
            return None
        row = rows[0]
        return MemoryRecord(
            id=row[0],
            text=row[1],
            created_at=row[2],
            importance=float(row[3]),
            access_count=int(row[4]),
            last_accessed=row[5],
            embedding=[float(value) for value in row[6]],
        )

    def clear_relationships(self, rel_name: str) -> None:
        self._conn.execute(f"MATCH ()-[r:{rel_name}]->() DELETE r")

    def create_similarity_pair(self, left_id: str, right_id: str, weight: float, updated_at: str) -> None:
        if left_id == right_id:
            return
        self._create_relationship("SIMILAR", left_id, right_id, weight, updated_at)
        self._create_relationship("SIMILAR", right_id, left_id, weight, updated_at)

    def create_next_edge(self, source_id: str, target_id: str, weight: float, updated_at: str) -> None:
        if source_id == target_id:
            return
        self._create_relationship("NEXT", source_id, target_id, weight, updated_at)

    def _create_relationship(
        self,
        rel_name: str,
        source_id: str,
        target_id: str,
        weight: float,
        updated_at: str,
    ) -> None:
        self._conn.execute(
            f"""
            MATCH (a:Memory {{id: $source_id}}), (b:Memory {{id: $target_id}})
            CREATE (a)-[:{rel_name} {{
                weight: $weight,
                updated_at: $updated_at
            }}]->(b)
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "weight": weight,
                "updated_at": updated_at,
            },
        )

    def list_similarity_edges(self) -> list[SimilarityEdge]:
        rows = self._conn.execute(
            """
            MATCH (a:Memory)-[r:SIMILAR]->(b:Memory)
            RETURN a.id, b.id, r.weight
            """
        ).get_all()
        return [
            SimilarityEdge(
                source_id=row[0],
                target_id=row[1],
                weight=float(row[2]),
            )
            for row in rows
        ]

    def list_next_edges(self) -> list[SimilarityEdge]:
        rows = self._conn.execute(
            """
            MATCH (a:Memory)-[r:NEXT]->(b:Memory)
            RETURN a.id, b.id, r.weight
            """
        ).get_all()
        return [
            SimilarityEdge(
                source_id=row[0],
                target_id=row[1],
                weight=float(row[2]),
            )
            for row in rows
        ]

    def touch_memories(self, memory_ids: list[str], timestamp: str) -> None:
        for memory_id in memory_ids:
            self._conn.execute(
                """
                MATCH (m:Memory {id: $id})
                SET m.access_count = m.access_count + 1,
                    m.last_accessed = $timestamp
                """,
                {"id": memory_id, "timestamp": timestamp},
            )

    def count_memories(self) -> int:
        rows = self._conn.execute("MATCH (m:Memory) RETURN count(m)").get_all()
        return int(rows[0][0]) if rows else 0

    def count_relationships(self, rel_name: str) -> int:
        rows = self._conn.execute(f"MATCH ()-[r:{rel_name}]->() RETURN count(r)").get_all()
        return int(rows[0][0]) if rows else 0
