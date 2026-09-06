"""Orchestrator + CLI: build all PyArrow tables in memory, bulk-load one .lbdb.

Pipeline (mirrors LdbcDatagen.scala GenerationStage):
  static (places/tagclasses/tags/orgs) -> persons -> knows -> forums -> messages
  -> open .lbdb -> CREATE schema -> bulk.load_nodes ->
  CREATE HASH INDEX on Person(ID) -> bulk.load_rels.
  (No ART/secondary indexes here; created separately if needed.)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from lbug_datagen.activity import generate_forums, generate_messages
from lbug_datagen import official as off
from lbug_datagen.bulk import load_nodes, load_rels
from lbug_datagen.dictionaries import Dictionaries
from lbug_datagen.knows import generate_knows
from lbug_datagen.params import DatagenConfig
from lbug_datagen.person import generate_persons
from lbug_datagen.schema import SCHEMA_DDL, secondary_cypher
from lbug_datagen.static_graph import generate_static


def build_tables(cfg: DatagenConfig, log=print) -> dict[str, pa.Table]:
    t0 = time.time()
    d = Dictionaries()
    o = off.get()
    static = generate_static(d)
    n_tags: int = static.pop("_tag_count")["n"][0].as_py()  # type: ignore
    city_ids = None
    if "_official_city_ids" in static:
        city_ids = static.pop("_official_city_ids")  # np array of place ids
        static.pop("_official_city_probs", None)
    else:
        city_ids = static.pop("_city_id")  # type: ignore
    log(f"static: {time.time()-t0:.2f}s")

    t0 = time.time()
    persons = generate_persons(cfg, d, city_ids, n_tags, cfg.seed, o=o)
    city_idx = persons.pop("_person_city")["city"].to_pylist()
    person_place = (persons.pop("_person_place")["place"].to_pylist()
                    if o is not None else None)
    creations = persons.pop("_person_creation")["creationDate"].to_pylist()
    log(f"persons ({cfg.num_persons}): {time.time()-t0:.2f}s")

    n_persons = cfg.num_persons
    if o is not None:
        # LSQB-calibrated knows pass mix (fewer correlated, more random edges)
        cfg.knows_percentages = (0.36, 0.36, 0.28)
    t0 = time.time()
    degree = None
    if o is not None:
        degree = np.clip(o.hists["knows_degree"].sample(
            np.random.default_rng(cfg.seed + 9), cfg.num_persons),
            0, cfg.num_persons - 1)
    target_pairs = (int(o.scalars["knows_undirected"] * n_persons
                        / o.scalars["n_persons"]) if o is not None else None)
    knows = generate_knows(cfg, n_persons, city_idx,
                           persons["hasInterest"], creations, cfg.seed,
                           log=log, workers=cfg.workers, degree=degree,
                           target_pairs=target_pairs)
    log(f"knows ({knows.num_rows}): {time.time()-t0:.2f}s")

    t0 = time.time()
    forums = generate_forums(cfg, d, cfg.num_persons, city_idx, creations,
                             persons["hasInterest"], cfg.seed,
                             o=o, n_tags=n_tags)
    n_forums: int = forums.pop("_forum_of")  # type: ignore
    forums.pop("_cont_placeholder", None)
    log(f"forums ({forums['Forum'].num_rows}): {time.time()-t0:.2f}s")

    t0 = time.time()
    friends = None
    if o is not None:
        kf = knows["FROM"].to_pylist()
        kt = knows["TO"].to_pylist()
        friends = {}
        for a, b in zip(kf, kt):
            friends.setdefault(a, []).append(b)  # knows is bidirectional
    msgs = generate_messages(cfg, d, cfg.num_persons, n_forums, creations,
                             forums["hasMember"], forums["forumHasTag"], cfg.seed,
                             workers=cfg.workers, off=o, n_tags=n_tags,
                             person_place=person_place, friends=friends)
    log(f"messages (posts={msgs['Post'].num_rows}, comments={msgs['Comment'].num_rows}): "
        f"{time.time()-t0:.2f}s")

    return {**static, **persons, "knows": knows, **forums, **msgs}


def write_lbdb(cfg: DatagenConfig, out: str, log=print) -> dict[str, int]:
    import ladybug as lb
    tables = build_tables(cfg, log)
    p = Path(out)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()
    # Remove sidecar artifacts from previous runs (WAL, journals, tmp files,
    # e.g. <out>.wal) so stale state can't interfere with the load.
    for sib in p.parent.glob(p.name + ".*"):
        if sib.is_dir():
            shutil.rmtree(sib)
        else:
            sib.unlink()
        log(f"removed stale artifact {sib}")
    db = lb.Database(out, auto_checkpoint=False,
                     checkpoint_threshold=1 << 60)  # checkpoint once at the end
    conn = lb.Connection(db)
    # Skip default PK hash indexes during bulk load (built/added later);
    # keeps ingestion to a pure bulk path without per-row index maintenance.
    conn.execute("CALL enable_default_hash_index = false;")
    conn.execute("CALL debug_enable_multi_writes=true;")
    for stmt in [s.strip() for s in SCHEMA_DDL.split(";") if s.strip()]:
        conn.execute(stmt)
    counts = load_nodes(conn, tables, cfg.arrow_chunk, log,
                        db=db, workers=cfg.workers)
    # Explicit PK hash index for Person (default hash indexes are disabled
    # above for bulk load). Speeds up rel-endpoint probes below and all later
    # PK lookups. ART/secondary indexes are intentionally NOT created here.
    t0 = time.time()
    conn.execute("CREATE HASH INDEX person_pk_hx FOR (n:Person) ON (n.ID)")
    log(f"hash index person_pk_hx: {time.time()-t0:.2f}s")
    counts.update(load_rels(conn, tables, cfg.arrow_chunk, log,
                            db=db, workers=cfg.workers))
    try:
        conn.execute("CHECKPOINT")
    except Exception as e:
        log(f"checkpoint: {e}")
    try:
        conn.close()
    except Exception:
        pass
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="LDBC SNB datagen → .lbdb (PyArrow bulk)")
    ap.add_argument("--scale-factor", default="0.003")
    ap.add_argument("--num-persons", type=int, default=0)
    ap.add_argument("--out", default="sf0.003.lbdb")
    ap.add_argument("--degree-mean", type=float, default=0.0,
                    help="mean knows-degree override (0 = reference formula)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1,
                    help="worker threads for concurrent bulk-load writes and "
                         "knows/messages generation")
    ap.add_argument("--secondary", action="store_true",
                    help="write secondary ART index cypher next to output")
    ap.add_argument("--arrow-chunk", type=int, default=200_000)
    a = ap.parse_args(argv)
    cfg = DatagenConfig(scale_factor=a.scale_factor, num_persons=a.num_persons,
                        seed=a.seed, arrow_chunk=a.arrow_chunk,
                        degree_mean=a.degree_mean, workers=a.workers).resolve()
    t0 = time.time()
    counts = write_lbdb(cfg, a.out)
    print(f"wrote {a.out} in {time.time()-t0:.1f}s; "
          f"persons={counts.get('Person', 0)} knows={counts.get('knows', 0)}")
    if a.secondary:
        sidecar = Path(a.out).with_suffix("")
        sidecar = sidecar.parent / (sidecar.name + ".secondary.cypher")
        sidecar.write_text(secondary_cypher())
        print(f"wrote {sidecar} (run separately for ART indexes)")


if __name__ == "__main__":
    main()
