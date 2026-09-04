# ldbc_snb_datagen_columnar

Python port of [ldbc_snb_datagen_spark](../ldbc_snb_datagen_spark) that writes
an LDBC SNB graph **directly to `.lbdb`** (LadybugDB) instead of CSV/Parquet.

Same algorithm, different sink:

| stage | spark datagen | this port |
|---|---|---|
| dictionaries | `src/main/resources/*` + `dictionary/*` | `dictionaries.py` (loads the same resource files) |
| params | `params_default.ini` + `scale_factors.xml` | `params.py` (same keys/defaults) |
| persons | `generators/PersonGenerator.java` | `person.py` |
| knows | `knowsgenerators/DistanceKnowsGenerator.java` (Facebook degree + geo/interest similarity, blocked sort) | `knows.py` |
| forums/messages | `ForumGenerator` + `PostGenerator`/`CommentGenerator`/`LikeGenerator` + `LdbcSnbTextGenerator` | `forum.py`, `messages.py` |
| static graph | `StaticOutputStream` (places, tagclasses, tags, orgs) | `static_graph.py` |
| sink | CSV/Parquet serializers | `bulk.py` — PyArrow tables in memory, **one bulk query per table** |

## Why PyArrow bulk load

LadybugDB is slow with row-at-a-time writes (`CREATE (p:Person ...)` per row
= one transaction per row). So every entity is first fully materialised as a
`pyarrow.Table` in memory, registered once with
`Connection.create_arrow_table(name, table)`, then copied with a **single**
Cypher statement per table:

```cypher
MATCH (n:arr_person) CREATE (p:Person {ID: n.ID, ...});
MATCH (n:arr_knows) MATCH (a:Person), (b:Person)
  WHERE a.ID = n.src AND b.ID = n.dst
  CREATE (a)-[e:knows {creationDate: n.creationDate}]->(b);
```

No per-row `execute()` calls. See `lbug_datagen/bulk.py`.

## Quick start

```bash
source .venv/bin/activate
uv pip install -e .          # or: .venv/bin/pip install -e .
lbug-datagen --scale-factor 0.003 --out sf0.003.lbdb --seed 42
lbug-datagen --scale-factor 1 --out sf1.lbdb --workers 16
lbug-datagen --help
```

`--workers N` parallelises knows (per-block) and messages (per-forum-range)
over a process pool. RNG is seeded per block/forum, so output is identical
for any worker count. Load path: nodes → explicit
`CREATE HASH INDEX person_pk_hx FOR (n:Person) ON (n.ID)` → rels.
No ART/secondary indexes are created inline; pass `--secondary` to also write
`<out-stem>.secondary.cypher` (e.g. `sf1.secondary.cypher`) with the
`CREATE ART INDEX` statements for a separate run.

## Layout

- `lbug_datagen/params.py` — `params_default.ini` defaults + `scale_factors.xml` person counts
- `lbug_datagen/distributions.py` — Facebook degree / power-law / date helpers
- `lbug_datagen/dictionaries.py` — loads spark `src/main/resources/*` (places, names, tags, …)
- `lbug_datagen/static_graph.py` — Place tree, TagClasses, Tags, Organisations
- `lbug_datagen/person.py` — correlated person attributes (location, birthday, names, study/work, interests)
- `lbug_datagen/knows.py` — distance/interest knows edges with target degrees
- `lbug_datagen/forum.py` — wall + group forums, membership, moderation
- `lbug_datagen/messages.py` — posts / comments / likes with tag-correlated text stubs
- `lbug_datagen/schema.py` — LadybugDB DDL (matches `ladybugdb/etl/schema.cypher`)
- `lbug_datagen/bulk.py` — Arrow → `.lbdb` bulk loader (the only writer)
- `lbug_datagen/generate.py` — orchestrator + CLI
