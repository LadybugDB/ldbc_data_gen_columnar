"""Forums + messages — ports of ForumGenerator / PostGenerator /
CommentGenerator / LikeGenerator + LdbcSnbTextGenerator (stub text).

Forums: one wall forum per person (moderator = owner) + group forums with
power-law membership; tags sampled from members' interests (flashmob skew).
Messages: per forum, posts over the simulation months (uniform + flashmob);
comments form reply trees on posts/comments; likes are Zipf over messages.
Text is a deterministic stub ("tag words + lorem") with matching length —
swap in the real Markov generator later without changing the loader.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa

from lbug_datagen.dictionaries import Dictionaries
from lbug_datagen.params import DatagenConfig

LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
         "tempor incididunt ut labore et dolore magna aliqua ").split()


def _text(rng: np.random.Generator, tag_words: list[str], length: int) -> str:
    words = list(tag_words) + LOREM
    out = []
    while sum(len(w) + 1 for w in out) < length:
        out.append(words[int(rng.integers(0, len(words)))])
    s = " ".join(out)[:length]
    return s


def generate_forums(cfg: DatagenConfig, d: Dictionaries, n: int,
                    city_idx: list[int], creations: list[int],
                    interests: pa.Table, seed: int) -> dict[str, pa.Table]:
    rng = np.random.default_rng(seed + 2)
    tag_of: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(interests["FROM"].to_pylist(), interests["TO"].to_pylist()):
        tag_of[a].append(b)

    f_id, f_title, f_date = [], [], []
    mod_from, mod_to = [], []
    mem_from, mem_to, mem_date = [], [], []
    ftag_from, ftag_to = [], []
    cont_from, cont_to = [], []

    fid = 0
    # wall forums: one per person
    for i in range(n):
        f_id.append(fid); f_title.append(f"Wall of Person{i}")
        f_date.append(int(creations[i])); mod_from.append(fid); mod_to.append(i)
        mem_from.append(fid); mem_to.append(i); mem_date.append(int(creations[i]))
        for t in tag_of[i][:3]:
            ftag_from.append(fid); ftag_to.append(t)
        fid += 1
    # group forums
    n_groups = max(1, n // 10)
    for g in range(n_groups):
        owner = int(rng.integers(0, n))
        f_id.append(fid); f_title.append(f"Group {g}")
        f_date.append(int(creations[owner])); mod_from.append(fid); mod_to.append(owner)
        size = int(rng.integers(2, min(cfg.max_num_member_group, n) + 1))
        members = rng.choice(n, size=min(size, n), replace=False)
        for m in members:
            m = int(m)
            mem_from.append(fid); mem_to.append(m)
            mem_date.append(int(max(creations[owner], creations[m])))
            for t in tag_of[m][:2]:
                if rng.random() < 0.5:
                    ftag_from.append(fid); ftag_to.append(t)
        fid += 1

    forums = pa.table({
        "ID": pa.array(f_id, type=pa.int64()),
        "title": f_title,
        "creationDate": pa.array(f_date, type=pa.timestamp("ms")),
    })
    out = {
        "Forum": forums,
        "hasModerator": pa.table({"FROM": mod_from, "TO": mod_to}),
        "hasMember": pa.table({"FROM": mem_from, "TO": mem_to,
                               "joinDate": pa.array(mem_date, type=pa.timestamp("ms"))}),
        "forumHasTag": pa.table({"FROM": ftag_from, "TO": ftag_to}),
        "_forum_of": fid,
    }
    # containerOf filled by messages.py
    out["_cont_placeholder"] = pa.table({"FROM": cont_from, "TO": cont_to})
    return out


# Shard context for worker processes (fork-inherited; tasks carry only ranges).
_MSG_CTX: dict = {}

# Shard-local list keys holding POST ids / COMMENT ids (rebased by parent).
_POST_ID_KEYS = ("post_ids", "post_creator_f", "post_loc_f", "post_tag_f",
                  "cont_t", "rpost_t", "like_p_t")
_COMMENT_ID_KEYS = ("c_ids", "c_creator_f", "c_loc_f", "c_tag_f",
                    "rpost_f", "rcomm_f", "rcomm_t", "like_c_t")
_MSG_LIST_KEYS = ("post_ids", "post_img", "post_cd", "post_ip", "post_br",
                   "post_lang", "post_ct", "post_len", "post_creator_f",
                   "post_creator_t", "post_loc_f", "post_loc_t", "post_tag_f",
                   "post_tag_t", "cont_f", "cont_t", "c_ids", "c_cd", "c_ip",
                   "c_br", "c_ct", "c_len", "c_creator_f", "c_creator_t",
                   "c_loc_f", "c_loc_t", "c_tag_f", "c_tag_t", "rpost_f",
                   "rpost_t", "rcomm_f", "rcomm_t", "like_c_f", "like_c_t",
                   "like_c_d", "like_p_f", "like_p_t", "like_p_d")


def _process_forum(f: int, members: list[int], ftags: list[int],
                   out: dict[str, list], pid: int, cid: int) -> tuple[int, int]:
    """Generate one forum's posts/comments/likes with shard-LOCAL ids.

    Per-forum seeding -> identical output for any worker count.
    """
    ctx = _MSG_CTX
    rng = np.random.default_rng(np.random.SeedSequence([ctx["seed"] + 3, f]))
    creations = ctx["creations"]
    n = ctx["n"]
    p0 = pid
    n_posts = int(rng.integers(0, 4))  # ~ up to a few posts per forum at tiny SF
    for _ in range(n_posts):
        author = members[int(rng.integers(0, len(members)))]
        cd = int(creations[author] + int(rng.integers(0, 300 * 86_400_000)))
        tag_words = [ctx["tag_names"][t % len(ctx["tag_names"])] for t in ftags[:2]]
        length = int(rng.integers(ctx["min_text_size"], ctx["max_text_size"] + 1))
        out["post_ids"].append(pid)
        out["post_img"].append("")
        out["post_cd"].append(cd)
        out["post_ip"].append(f"2.3.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}")
        out["post_br"].append(ctx["browsers"][int(rng.integers(0, len(ctx["browsers"])))])
        out["post_lang"].append("English" if rng.random() < ctx["prob_english"] else
                         ctx["languages"][int(rng.integers(0, len(ctx["languages"])))])
        out["post_ct"].append(_text(rng, tag_words, length))
        out["post_len"].append(length)
        out["post_creator_f"].append(pid); out["post_creator_t"].append(author)
        out["post_loc_f"].append(pid); out["post_loc_t"].append(0)
        for t in ftags[:2]:
            out["post_tag_f"].append(pid); out["post_tag_t"].append(t)
        out["cont_f"].append(f); out["cont_t"].append(pid)
        pid += 1
    # comments/likes only on THIS forum's posts (local ids p0..pid).
    post_cd_of = (dict(zip(range(p0, pid), out["post_cd"][len(out["post_cd"]) - (pid - p0):]))
                  if pid > p0 else {})
    for p in range(p0, pid):
        for _ in range(int(rng.integers(0, ctx["max_num_comments"] // 4 + 1))):
            replier = int(rng.integers(0, n))
            cd = int(post_cd_of[p] + int(rng.integers(0, 10 * 86_400_000)))
            length = int(rng.integers(50, 200))
            out["c_ids"].append(cid); out["c_cd"].append(cd)
            out["c_ip"].append(f"3.4.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}")
            out["c_br"].append(ctx["browsers"][int(rng.integers(0, len(ctx["browsers"])))])
            out["c_ct"].append(_text(rng, ["reply"], length)); out["c_len"].append(length)
            out["c_creator_f"].append(cid); out["c_creator_t"].append(replier)
            out["c_loc_f"].append(cid); out["c_loc_t"].append(0)
            out["rpost_f"].append(cid); out["rpost_t"].append(p)
            if rng.random() < 0.3:
                liker = int(rng.integers(0, n))
                out["like_c_f"].append(liker); out["like_c_t"].append(cid)
                out["like_c_d"].append(int(cd + int(rng.integers(0, 5 * 86_400_000))))
            cid += 1
    for p in range(p0, pid):
        for _ in range(int(rng.integers(0, 3))):
            liker = int(rng.integers(0, n))
            out["like_p_f"].append(liker); out["like_p_t"].append(p)
            out["like_p_d"].append(int(post_cd_of[p] + int(rng.integers(0, 5 * 86_400_000))))
    return pid, cid


def _messages_shard(task) -> tuple[int, dict[str, list], int, int]:
    """Process a contiguous forum range. Top-level (picklable)."""
    shard_idx, f0, f1 = task
    ctx = _MSG_CTX
    members_of = ctx["members_of"]
    tags_of_forum = ctx["tags_of_forum"]
    out = {k: [] for k in _MSG_LIST_KEYS}
    pid = cid = 0
    for f in range(f0, f1):
        members = members_of.get(f)
        if not members:
            continue
        pid, cid = _process_forum(f, members, tags_of_forum.get(f, [0]), out, pid, cid)
    return shard_idx, out, pid, cid


def _merge_shards(shards: list[tuple[int, dict[str, list], int, int]]
                    ) -> dict[str, list]:
    """Merge shard-local lists to global ids (prefix-sum rebase)."""
    shards = sorted(shards, key=lambda s: s[0])
    master = {k: [] for k in _MSG_LIST_KEYS}
    post_off = comm_off = 0
    for _, out, n_posts, n_comments in shards:
        for k, vals in out.items():
            if k in _POST_ID_KEYS:
                master[k].extend(v + post_off for v in vals)
            elif k in _COMMENT_ID_KEYS:
                master[k].extend(v + comm_off for v in vals)
            else:
                master[k].extend(vals)
        post_off += n_posts
        comm_off += n_comments
    return master


def _build_message_tables(m: dict[str, list]) -> dict[str, pa.Table]:
    def ts(vals):
        return pa.array(vals, type=pa.timestamp("ms"))

    def ids(vals):
        return pa.array(vals, type=pa.int64())

    posts = pa.table({
        "ID": ids(m["post_ids"]),
        "imageFile": m["post_img"],
        "creationDate": ts(m["post_cd"]),
        "locationIP": m["post_ip"], "browserUsed": m["post_br"],
        "language": m["post_lang"],
        "content": m["post_ct"], "length": ids(m["post_len"]),
    })
    comments = pa.table({
        "ID": ids(m["c_ids"]),
        "creationDate": ts(m["c_cd"]),
        "locationIP": m["c_ip"], "browserUsed": m["c_br"], "content": m["c_ct"],
        "length": ids(m["c_len"]),
    })
    return {
        "Post": posts,
        "Comment": comments,
        "postHasCreator": pa.table({"FROM": ids(m["post_creator_f"]),
                                     "TO": ids(m["post_creator_t"])}),
        "commentHasCreator": pa.table({"FROM": ids(m["c_creator_f"]),
                                        "TO": ids(m["c_creator_t"])}),
        "containerOf": pa.table({"FROM": ids(m["cont_f"]), "TO": ids(m["cont_t"])}),
        "postIsLocatedIn": pa.table({"FROM": ids(m["post_loc_f"]),
                                      "TO": ids(m["post_loc_t"])}),
        "commentIsLocatedIn": pa.table({"FROM": ids(m["c_loc_f"]),
                                         "TO": ids(m["c_loc_t"])}),
        "postHasTag": pa.table({"FROM": ids(m["post_tag_f"]),
                                 "TO": ids(m["post_tag_t"])}),
        "commentHasTag": pa.table({"FROM": ids(m["c_tag_f"]), "TO": ids(m["c_tag_t"])}),
        "replyOfPost": pa.table({"FROM": ids(m["rpost_f"]), "TO": ids(m["rpost_t"])}),
        "replyOfComment": pa.table({"FROM": ids(m["rcomm_f"]), "TO": ids(m["rcomm_t"])}),
        "likePost": pa.table({"FROM": ids(m["like_p_f"]), "TO": ids(m["like_p_t"]),
                              "creationDate": ts(m["like_p_d"])}),
        "likeComment": pa.table({"FROM": ids(m["like_c_f"]), "TO": ids(m["like_c_t"]),
                                 "creationDate": ts(m["like_c_d"])}),
    }


def generate_messages(cfg: DatagenConfig, d: Dictionaries, n: int,
                      n_forums: int, creations: list[int],
                      forum_members: pa.Table, forum_tags: pa.Table,
                      seed: int, workers: int = 1) -> dict[str, pa.Table]:
    global _MSG_CTX
    mem_f = forum_members["FROM"].to_pylist()
    mem_p = forum_members["TO"].to_pylist()
    members_of: dict[int, list[int]] = {}
    for f, p in zip(mem_f, mem_p):
        members_of.setdefault(f, []).append(p)
    tag_f = forum_tags["FROM"].to_pylist()
    tag_t = forum_tags["TO"].to_pylist()
    tags_of_forum: dict[int, list[int]] = {}
    for f, t in zip(tag_f, tag_t):
        tags_of_forum.setdefault(f, []).append(t)
    _MSG_CTX = {
        "seed": seed, "n": n, "creations": list(creations),
        "members_of": members_of, "tags_of_forum": tags_of_forum,
        "tag_names": list(d.tag_names), "browsers": list(d.browsers),
        "languages": list(d.languages),
        "min_text_size": cfg.min_text_size, "max_text_size": cfg.max_text_size,
        "max_num_comments": cfg.max_num_comments, "prob_english": cfg.prob_english,
    }
    # Contiguous forum ranges; per-forum seeding => worker-invariant output.
    tasks = []
    if n_forums > 0:
        span = (n_forums + max(workers, 1) - 1) // max(workers, 1)
        for w in range(max(workers, 1)):
            f0, f1 = w * span, min((w + 1) * span, n_forums)
            if f0 < f1:
                tasks.append((w, f0, f1))
    if workers <= 1:
        shards = [_messages_shard(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            shards = list(ex.map(_messages_shard, tasks))
    merged = _merge_shards(shards)
    return _build_message_tables(merged)
