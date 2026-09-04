"""Bulk loader: PyArrow (in memory) -> .lbdb with ONE query per chunk.

LadybugDB is slow with small (row-at-a-time) writes, so we NEVER do per-row
``conn.execute(CREATE ...)``. Instead, for each entity:

1. the generator builds a complete ``pyarrow.Table`` in memory;
2. we register it via ``Connection.create_arrow_table(tmp_name, table)``;
3. we run ONE Cypher statement that copies tmp -> real table.

Nodes:  ``MATCH (n:tmp) CREATE (p:Real {...})``
Rels:   ``MATCH (n:tmp) MATCH (a:Src),(b:Dst)
         WHERE a.ID = n.FROM AND b.ID = n.TO CREATE (a)-[e:Rel {...}]->(b)``

Large tables are split into ``chunk``-row slices so one Arrow registration
never blows up RAM. Each chunk is still a single query (bulk).

DATE columns: generators emit ISO date strings; the loader casts with
``date(n.birthday)`` — still inside the bulk query. Empty tables are skipped.
"""
from __future__ import annotations

import pyarrow as pa

# node table -> [(real_prop, arrow_col, cast)] ; cast wraps the expression.
NODE_COPIES: dict[str, list[tuple[str, str, str]]] = {
    "Person": [("ID", "ID", "{e}"), ("firstName", "firstName", "{e}"),
               ("lastName", "lastName", "{e}"), ("gender", "gender", "{e}"),
               ("birthday", "birthday", "date({e})"),
               ("creationDate", "creationDate", "{e}"),
               ("locationIP", "locationIP", "{e}"),
               ("browserUsed", "browserUsed", "{e}")],
    "Place": [("ID", "ID", "{e}"), ("name", "name", "{e}"),
              ("url", "url", "{e}"), ("type", "type", "{e}")],
    "Organisation": [("ID", "ID", "{e}"), ("type", "type", "{e}"),
                     ("name", "name", "{e}"), ("url", "url", "{e}")],
    "Tag": [("ID", "ID", "{e}"), ("name", "name", "{e}"), ("url", "url", "{e}")],
    "Tagclass": [("ID", "ID", "{e}"), ("name", "name", "{e}"), ("url", "url", "{e}")],
    "Forum": [("ID", "ID", "{e}"), ("title", "title", "{e}"),
              ("creationDate", "creationDate", "{e}")],
    "Post": [("ID", "ID", "{e}"), ("imageFile", "imageFile", "{e}"),
             ("creationDate", "creationDate", "{e}"),
             ("locationIP", "locationIP", "{e}"),
             ("browserUsed", "browserUsed", "{e}"),
             ("language", "language", "{e}"), ("content", "content", "{e}"),
             ("length", "length", "{e}")],
    "Comment": [("ID", "ID", "{e}"), ("creationDate", "creationDate", "{e}"),
                ("locationIP", "locationIP", "{e}"),
                ("browserUsed", "browserUsed", "{e}"),
                ("content", "content", "{e}"), ("length", "length", "{e}")],
}

# rel table -> (from_node, to_node, [(real_prop, arrow_col)])
REL_COPIES: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "knows": ("Person", "Person", [("creationDate", "creationDate")]),
    "hasMember": ("Forum", "Person", [("joinDate", "joinDate")]),
    "likePost": ("Person", "Post", [("creationDate", "creationDate")]),
    "likeComment": ("Person", "Comment", [("creationDate", "creationDate")]),
    "studyAt": ("Person", "Organisation", [("classYear", "classYear")]),
    "workAt": ("Person", "Organisation", [("workFrom", "workFrom")]),
    "containerOf": ("Forum", "Post", []),
    "commentHasCreator": ("Comment", "Person", []),
    "postHasCreator": ("Post", "Person", []),
    "hasInterest": ("Person", "Tag", []),
    "hasModerator": ("Forum", "Person", []),
    "commentHasTag": ("Comment", "Tag", []),
    "forumHasTag": ("Forum", "Tag", []),
    "postHasTag": ("Post", "Tag", []),
    "hasType": ("Tag", "Tagclass", []),
    "commentIsLocatedIn": ("Comment", "Place", []),
    "organisationIsLocatedIn": ("Organisation", "Place", []),
    "personIsLocatedIn": ("Person", "Place", []),
    "postIsLocatedIn": ("Post", "Place", []),
    "isPartOf": ("Place", "Place", []),
    "isSubclassOf": ("Tagclass", "Tagclass", []),
    "replyOfComment": ("Comment", "Comment", []),
    "replyOfPost": ("Comment", "Post", []),
}

NODE_ORDER = ["Place", "Tagclass", "Tag", "Organisation",
              "Person", "Forum", "Post", "Comment"]
REL_ORDER = ["isPartOf", "isSubclassOf", "hasType", "organisationIsLocatedIn",
             "personIsLocatedIn", "studyAt", "workAt", "hasInterest", "knows",
             "hasModerator", "hasMember", "forumHasTag", "containerOf",
             "postHasCreator", "postIsLocatedIn", "postHasTag",
             "commentHasCreator", "commentIsLocatedIn", "commentHasTag",
             "replyOfPost", "replyOfComment", "likePost", "likeComment"]


def _chunks(table: pa.Table, size: int):
    if size <= 0 or table.num_rows <= size:
        yield table
        return
    for off in range(0, table.num_rows, size):
        yield table.slice(off, min(size, table.num_rows - off))


def bulk_copy_nodes(conn, arrow_name: str, real_table: str,
                    table: pa.Table, chunk: int = 200_000) -> int:
    """Bulk-copy one Arrow node table into ``real_table``. Returns row count."""
    if table.num_rows == 0:
        return 0
    mapping = NODE_COPIES[real_table]
    props = ", ".join(
        f"{prop}: {cast.format(e=f'n.{col}')}" for prop, col, cast in mapping)
    total = 0
    single = table.num_rows <= chunk or chunk <= 0
    for i, part in enumerate(_chunks(table, chunk)):
        tmp = arrow_name if single else f"{arrow_name}_{i}"
        conn.create_arrow_table(tmp, part)
        try:
            conn.execute(f"MATCH (n:{tmp}) CREATE (p:{real_table} {{{props}}})")
        finally:
            try:
                conn.drop_arrow_table(tmp)
            except Exception:
                pass
        total += part.num_rows
    return total


def bulk_copy_rels(conn, arrow_name: str, real_table: str,
                   table: pa.Table, chunk: int = 200_000) -> int:
    """Bulk-copy one Arrow edge table into ``real_table``. Returns row count."""
    if table.num_rows == 0:
        return 0
    src_t, dst_t, props = REL_COPIES[real_table]
    prop_str = (" {" + ", ".join(f"{p}: n.{c}" for p, c in props) + "}"
                if props else "")
    total = 0
    single = table.num_rows <= chunk or chunk <= 0
    for i, part in enumerate(_chunks(table, chunk)):
        tmp = arrow_name if single else f"{arrow_name}_{i}"
        conn.create_arrow_table(tmp, part)
        try:
            conn.execute(
                f"MATCH (n:{tmp}) MATCH (a:{src_t}), (b:{dst_t}) "
                f"WHERE a.ID = n.FROM AND b.ID = n.TO "
                f"CREATE (a)-[e:{real_table}{prop_str}]->(b)")
        finally:
            try:
                conn.drop_arrow_table(tmp)
            except Exception:
                pass
        total += part.num_rows
    return total


def load_nodes(conn, tables: dict[str, pa.Table], chunk: int = 200_000,
               log=print) -> dict[str, int]:
    """Bulk-load node tables (FK order). No small writes."""
    counts: dict[str, int] = {}
    for t in NODE_ORDER:
        if t in tables:
            n = bulk_copy_nodes(conn, f"arr_{t}", t, tables[t], chunk)
            counts[t] = n
            log(f"nodes {t}: {n}")
    return counts


def load_rels(conn, tables: dict[str, pa.Table], chunk: int = 200_000,
              log=print) -> dict[str, int]:
    """Bulk-load relationship tables. No small writes."""
    counts: dict[str, int] = {}
    for t in REL_ORDER:
        if t in tables and tables[t].num_rows:
            n = bulk_copy_rels(conn, f"arr_{t}", t, tables[t], chunk)
            counts[t] = n
            log(f"rels {t}: {n}")
    return counts


def load_all(conn, tables: dict[str, pa.Table], chunk: int = 200_000,
             log=print) -> dict[str, int]:
    """Bulk-load every entity; nodes first (FK order), then rels. No small writes."""
    return {**load_nodes(conn, tables, chunk, log),
            **load_rels(conn, tables, chunk, log)}
