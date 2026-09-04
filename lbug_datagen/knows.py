"""Knows generation — port of DistanceKnowsGenerator + Knows + GenerationStage.

Reference algorithm (Spark):
1. Each person gets a degree budget ``maxNumKnows`` from FacebookDegreeDistribution
   (bucketed FB crawl → mean = round(N^(0.512-0.028*log10(N))), ~41 at SF1).
2. Three passes fill slices of that budget (GenerationStage percentages):
   45% university/country-correlated, 45% interest-correlated, 10% random.
   Each pass re-ranks persons (SparkRanker) and chunks the ranking into
   ``blockSize`` blocks; friendships form only within a block.
3. Within a block, persons fill up in rank order: person ``i`` scans ``j>i``
   and links with probability ``baseProbCorrelated^dist`` (0.95^dist), floored
   by ``limitProCorrelated`` (0.2). Either endpoint full → skip.
4. The RAW/COMPOSITE serializers emit each undirected friendship ONCE
   (verified: zero reverse duplicates in 40k sampled SF1 CSV rows), so we emit
   one row per pair. Cross-pass duplicates are resolved post-hoc with
   ``np.unique`` — the analogue of Spark's ``FriendshipMerger``.

Parallelism (``--workers``): blocks are independent by construction (edges never
cross block boundaries), so each pass maps its blocks over a process pool.
RNG is seeded per (seed, pass, block-index) via SeedSequence, therefore output
is IDENTICAL for any worker count. Tuning knobs: ``degree_mean``,
``base_prob_correlated``, ``limit_prob_correlated``, ``knows_percentages``.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

from lbug_datagen.distributions import facebook_degree
from lbug_datagen.params import DatagenConfig

MS_DAY = 86_400_000


def _pass_budgets(pcts: list[float], degree: np.ndarray, upto: int) -> np.ndarray:
    """Knows.targetEdges: NEW edges allowed for pass ``upto`` (0-based)."""
    d = degree.astype(np.int64)
    generated = np.zeros_like(d)
    for t in range(upto):
        generated = generated + np.ceil(np.asarray(pcts[t]) * d).astype(np.int64)
    generated = np.minimum(generated, d)
    return np.minimum(d - generated,
                      np.ceil(np.asarray(pcts[upto]) * d).astype(np.int64))


def _knows_block_task(task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pairs for ONE block of one pass. Top-level (picklable).

    Task: (block_ids, want, degree, creat, base, floor, seed_words).
    All arrays are block-aligned; ``j`` always refers inside the block.
    Returns (src, dst, dates) int64 arrays with a < b (may contain duplicates
    across passes — resolved by the parent, like FriendshipMerger).
    """
    block, want_b, degree_b, creat_b, base, floor, seed_words = task
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    B = len(block)
    have_b = np.zeros(B, dtype=np.int64)
    adj_b = [set() for _ in range(B)]
    out_a, out_b, out_d = [], [], []
    logb = math.log(base)
    for ii in range(B):
        if have_b[ii] >= want_b[ii] or have_b[ii] >= degree_b[ii]:
            continue
        ntail = B - ii - 1
        if ntail <= 0:
            continue
        dist = np.arange(1, ntail + 1, dtype=np.float64)
        prob = np.exp(dist * logb)
        u = rng.random(ntail)
        gi = int(block[ii])
        for k in range(ntail):
            if have_b[ii] >= want_b[ii] or have_b[ii] >= degree_b[ii]:
                break
            uk = u[k]
            if not (uk < prob[k] or uk < floor):
                continue
            jj = ii + 1 + k
            if have_b[jj] >= degree_b[jj]:
                continue
            gj = int(block[jj])
            if gj in adj_b[ii]:
                continue
            adj_b[ii].add(gj)
            adj_b[jj].add(gi)
            have_b[ii] += 1
            have_b[jj] += 1
            a, b = (gi, gj) if gi < gj else (gj, gi)
            m = creat_b[ii] if creat_b[ii] >= creat_b[jj] else creat_b[jj]
            out_a.append(a)
            out_b.append(b)
            out_d.append(int(m + int(rng.integers(0, 30 * MS_DAY))))
    return (np.asarray(out_a, dtype=np.int64),
            np.asarray(out_b, dtype=np.int64),
            np.asarray(out_d, dtype=np.int64))


def generate_knows(cfg: DatagenConfig, n: int, city_idx: list[int],
                   interests: pa.Table, creations: list[int],
                   seed: int, log=print, workers: int = 1) -> pa.Table:
    rng = np.random.default_rng(seed + 1)
    degree = facebook_degree(rng, n, num_persons=cfg.num_persons or n,
                             mean_override=cfg.degree_mean)
    degree = np.clip(degree, 0,
                     min(cfg.max_num_friends, max(n - 1, 1))).astype(np.int64)

    city_idx_list = list(city_idx)
    int_from = interests["FROM"].to_pylist()
    int_to = interests["TO"].to_pylist()
    main_tag = np.full(n, -1, dtype=np.int64)
    for a, b in zip(int_from, int_to):
        if main_tag[a] == -1:
            main_tag[a] = b
    creat = np.asarray(list(creations), dtype=np.int64)

    pcts = list(cfg.knows_percentages)
    base = cfg.base_prob_correlated
    floor = cfg.limit_prob_correlated

    orders = [
        np.array(sorted(range(n), key=lambda i: (city_idx_list[i], i)), dtype=np.int64),
        np.array(sorted(range(n),
                        key=lambda i: (int(main_tag[i]), city_idx_list[i], i)),
                 dtype=np.int64),
        np.random.default_rng(seed + 11).permutation(n).astype(np.int64),
    ]

    have = np.zeros(n, dtype=np.int64)
    all_src, all_dst, all_dates = [], [], []
    for s, order in enumerate(orders):
        budget = _pass_budgets(pcts, degree, s)
        want = have + budget
        tasks = []
        for bi, b0 in enumerate(range(0, n, cfg.block_size)):
            blk = order[b0:b0 + cfg.block_size]
            tasks.append((blk, want[blk], degree[blk], creat[blk],
                          base, floor, (seed + 1, s, bi)))
        if workers <= 1:
            results = [_knows_block_task(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_knows_block_task, tasks))
        for src, dst, dates in results:
            if len(src) == 0:
                continue
            np.add.at(have, src, 1)   # pre-merger sizes, FriendshipMerger-style
            np.add.at(have, dst, 1)
            all_src.append(src)
            all_dst.append(dst)
            all_dates.append(dates)
        log(f"knows pass {s}: tasks={len(tasks)}")

    if not all_src:
        empty = np.zeros(0, dtype=np.int64)
        return pa.table({"FROM": empty, "TO": empty,
                         "creationDate": pa.array([], type=pa.timestamp("ms"))})
    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    dates = np.concatenate(all_dates)
    keys = (src << np.int64(32)) | dst
    _, first_idx = np.unique(keys, return_index=True)  # FriendshipMerger analogue
    first_idx.sort()  # stable, deterministic order
    return pa.table({
        "FROM": pa.array(src[first_idx], type=pa.int64()),
        "TO": pa.array(dst[first_idx], type=pa.int64()),
        "creationDate": pa.array(dates[first_idx], type=pa.timestamp("ms")),
    })
