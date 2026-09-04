"""Parallel generation (workers) + explicit Person PK hash index."""
import pyarrow as pa

from lbug_datagen.activity import generate_forums, generate_messages
from lbug_datagen.bulk import load_nodes, load_rels
from lbug_datagen.dictionaries import Dictionaries
from lbug_datagen.knows import generate_knows
from lbug_datagen.params import DatagenConfig
from lbug_datagen.schema import SCHEMA_DDL


def _small_inputs(n=60, seed=11):
    cfg = DatagenConfig(scale_factor="0.003", seed=seed)
    cfg.num_persons = n
    cfg.block_size = 25  # force multiple blocks
    d = Dictionaries()
    from lbug_datagen.person import generate_persons
    from lbug_datagen.static_graph import generate_static
    static = generate_static(d)
    city_ids = static.pop("_city_id")
    n_tags = static.pop("_tag_count")["n"][0].as_py()
    persons = generate_persons(cfg, d, city_ids, n_tags, seed)
    city_idx = persons.pop("_person_city")["city"].to_pylist()
    creations = persons.pop("_person_creation")["creationDate"].to_pylist()
    return cfg, d, persons, city_idx, creations


def _sorted(t: pa.Table) -> pa.Table:
    cols = t.column_names
    if t.num_rows == 0:
        return t
    if "FROM" in cols and "TO" in cols:
        idx = sorted(range(t.num_rows),
                     key=lambda i: (t["FROM"][i].as_py(), t["TO"][i].as_py()))
        return t.take(idx)
    return t


def test_knows_worker_invariant():
    cfg, _, persons, city_idx, creations = _small_inputs()
    k1 = generate_knows(cfg, cfg.num_persons, city_idx, persons["hasInterest"],
                        creations, cfg.seed, log=lambda *a: None, workers=1)
    k2 = generate_knows(cfg, cfg.num_persons, city_idx, persons["hasInterest"],
                        creations, cfg.seed, log=lambda *a: None, workers=2)
    assert k1.num_rows > 0
    assert _sorted(k1).equals(_sorted(k2))


def test_messages_worker_invariant_and_valid():
    cfg, d, persons, city_idx, creations = _small_inputs()
    forums = generate_forums(cfg, d, cfg.num_persons, city_idx, creations,
                             persons["hasInterest"], cfg.seed)
    n_forums = forums.pop("_forum_of")
    forums.pop("_cont_placeholder", None)
    m1 = generate_messages(cfg, d, cfg.num_persons, n_forums, creations,
                           forums["hasMember"], forums["forumHasTag"],
                           cfg.seed, workers=1)
    m3 = generate_messages(cfg, d, cfg.num_persons, n_forums, creations,
                           forums["hasMember"], forums["forumHasTag"],
                           cfg.seed, workers=3)
    for k in m1:
        assert _sorted(m1[k]).equals(_sorted(m3[k])), k
    np_ = m1["Post"].num_rows
    nc = m1["Comment"].num_rows
    for tbl, col, bound in [("replyOfPost", "TO", np_), ("likePost", "TO", np_),
                            ("containerOf", "TO", np_),
                            ("postHasCreator", "FROM", np_),
                            ("replyOfPost", "FROM", nc),
                            ("commentHasCreator", "FROM", nc)]:
        vals = m1[tbl][col].to_pylist()
        assert all(0 <= v < bound for v in vals), (tbl, col)


def test_secondary_flag_writes_runnable_cypher(tmp_path):
    from lbug_datagen.generate import main
    out = str(tmp_path / "tiny.lbdb")
    main(["--num-persons", "20", "--seed", "5", "--out", out, "--secondary"])
    sidecar = tmp_path / "tiny.secondary.cypher"
    assert sidecar.exists()
    cypher = sidecar.read_text()
    assert "CREATE ART INDEX Person_firstName_art" in cypher
    import ladybug as lb
    conn = lb.Connection(lb.Database(out))
    for stmt in [s.strip() for s in cypher.split(";") if s.strip()]:
        conn.execute(stmt)
    idx = conn.execute("CALL show_indexes() RETURN *").get_as_df().to_string()
    assert "Person_firstName_art" in idx


def test_stale_artifacts_removed(tmp_path):
    from lbug_datagen.generate import write_lbdb
    from lbug_datagen.params import DatagenConfig
    out = tmp_path / "tiny.lbdb"
    out.write_bytes(b"stale-db")
    (tmp_path / "tiny.lbdb.wal").write_bytes(b"stale-wal")
    (tmp_path / "tiny.lbdb.tmp").write_bytes(b"stale-tmp")
    keep = tmp_path / "tiny.secondary.cypher"
    keep.write_text("keep me")
    cfg = DatagenConfig(scale_factor="0.003", seed=5)
    cfg.num_persons = 20
    write_lbdb(cfg, str(out), log=lambda *a: None)
    assert out.exists()
    assert not (tmp_path / "tiny.lbdb.wal").exists()
    assert not (tmp_path / "tiny.lbdb.tmp").exists()
    assert keep.read_text() == "keep me"
    import ladybug as lb
    conn = lb.Connection(lb.Database(str(out)))
    assert conn.execute("MATCH (p:Person) RETURN count(*)").get_as_df().iloc[0, 0] == 20


def test_person_pk_hash_index(tmp_path):
    import ladybug as lb
    db = lb.Database(str(tmp_path / "hx.lbdb"))
    conn = lb.Connection(db)
    conn.execute("CALL enable_default_hash_index = false;")
    for stmt in [s.strip() for s in SCHEMA_DDL.split(";") if s.strip()]:
        conn.execute(stmt)
    tables = {
        "Person": pa.table({"ID": [1, 2, 3],
                            "firstName": ["a", "b", "c"],
                            "lastName": ["x", "y", "z"],
                            "gender": ["male", "female", "male"],
                            "birthday": ["1990-01-01"] * 3,
                            "creationDate": pa.array([1710000000000] * 3,
                                                     type=pa.timestamp("ms")),
                            "locationIP": ["1.1.1.1"] * 3,
                            "browserUsed": ["Chrome"] * 3}),
        "knows": pa.table({"FROM": [1, 2], "TO": [2, 3],
                           "creationDate": pa.array([1710000000000] * 2,
                                                    type=pa.timestamp("ms"))}),
    }
    load_nodes(conn, tables, log=lambda *a: None)
    conn.execute("CREATE HASH INDEX person_pk_hx FOR (n:Person) ON (n.ID)")
    load_rels(conn, tables, log=lambda *a: None)
    assert conn.execute("MATCH (p:Person) RETURN count(*)").get_as_df().iloc[0, 0] == 3
    assert conn.execute("MATCH (a)-[e:knows]->(b) RETURN count(*)").get_as_df().iloc[0, 0] == 2
    idx = conn.execute("CALL show_indexes() RETURN *").get_as_df()
    assert "person_pk_hx" in idx.to_string()
