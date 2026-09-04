"""Smoke test: sf0.003-sized in-memory-equivalent build + bulk load to tmp .lbdb."""
import pyarrow as pa

from lbug_datagen.activity import generate_forums, generate_messages
from lbug_datagen.bulk import load_all
from lbug_datagen.dictionaries import Dictionaries
from lbug_datagen.knows import generate_knows
from lbug_datagen.params import DatagenConfig
from lbug_datagen.person import generate_persons
from lbug_datagen.schema import SCHEMA_DDL
from lbug_datagen.static_graph import generate_static


def test_smoke(tmp_path):
    import ladybug as lb
    cfg = DatagenConfig(scale_factor="0.003", seed=7).resolve()
    cfg.num_persons = 20  # keep the test fast
    d = Dictionaries()
    static = generate_static(d)
    city_ids = static.pop("_city_id")
    n_tags = static.pop("_tag_count")["n"][0].as_py()
    persons = generate_persons(cfg, d, city_ids, n_tags, cfg.seed)
    city_idx = persons.pop("_person_city")["city"].to_pylist()
    creations = persons.pop("_person_creation")["creationDate"].to_pylist()
    knows = generate_knows(cfg, cfg.num_persons, city_idx,
                           persons["hasInterest"], creations, cfg.seed)
    forums = generate_forums(cfg, d, cfg.num_persons, city_idx, creations,
                             persons["hasInterest"], cfg.seed)
    n_forums = forums.pop("_forum_of")
    forums.pop("_cont_placeholder", None)
    msgs = generate_messages(cfg, d, cfg.num_persons, n_forums, creations,
                             forums["hasMember"], forums["forumHasTag"], cfg.seed)
    tables = {**static, **persons, "knows": knows, **forums, **msgs}

    db = lb.Database(str(tmp_path / "smoke.lbdb"))
    conn = lb.Connection(db)
    for stmt in [s.strip() for s in SCHEMA_DDL.split(";") if s.strip()]:
        conn.execute(stmt)
    counts = load_all(conn, tables, log=lambda *a: None)
    assert counts["Person"] == 20
    assert counts["knows"] == knows.num_rows
    got = conn.execute("MATCH (p:Person) RETURN count(*)").get_as_df().iloc[0, 0]
    assert int(got) == 20
    got_k = conn.execute("MATCH (a)-[e:knows]->(b) RETURN count(*)").get_as_df().iloc[0, 0]
    assert int(got_k) == knows.num_rows
