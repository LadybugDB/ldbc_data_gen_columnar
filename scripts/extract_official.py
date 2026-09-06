"""Extract official LSQB sf1 distributions + static graph from sf1-official.lbdb.

Writes to lbug_datagen/resources/official/:
  place.csv, isPartOf.csv, tag.csv, hasType.csv, tagclass.csv, isSubclassOf.csv,
  organisation.csv, orgLocatedIn.csv, citypop.csv, dists.json

The generator (lbug_datagen/official.py) samples from these when present so the
generated graph matches the official LSQB sf1 distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import ladybug

OUT = Path(__file__).resolve().parent.parent / "lbug_datagen" / "resources" / "official"
DB = Path(__file__).resolve().parent.parent / "sf1-official.lbdb"


def rows(con, q):
    r = con.execute(q)
    out = []
    while r.has_next():
        out.append(list(r.get_next()))
    return out


def write_csv(path: Path, header: list[str], rows: list[list]):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    con = ladybug.Connection(ladybug.Database(str(DB), read_only=True))
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- static graph (ids + structure only; names are irrelevant to LSQB) ----
    write_csv(OUT / "place.csv", ["ID", "type"],
              rows(con, "MATCH (p:Continent) RETURN p.ContinentId, 'Continent' ORDER BY p.ContinentId")
              + rows(con, "MATCH (p:Country) RETURN p.CountryId, 'Country' ORDER BY p.CountryId")
              + rows(con, "MATCH (p:City) RETURN p.CityId, 'City' ORDER BY p.CityId"))
    write_csv(OUT / "isPartOf.csv", ["FROM", "TO"],
              rows(con, "MATCH (a:City)-[e:City_isPartOf_Country]->(b:Country) RETURN a.CityId, b.CountryId"))
    write_csv(OUT / "tagclass.csv", ["ID"],
              rows(con, "MATCH (t:TagClass) RETURN t.TagClassId ORDER BY t.TagClassId"))
    write_csv(OUT / "isSubclassOf.csv", ["FROM", "TO"],
              rows(con, "MATCH (a:TagClass)-[e:TagClass_isSubclassOf_TagClass]->(b:TagClass) RETURN a.TagClassId, b.TagClassId"))
    write_csv(OUT / "tag.csv", ["ID"],
              rows(con, "MATCH (t:Tag) RETURN t.TagId ORDER BY t.TagId"))
    write_csv(OUT / "hasType.csv", ["FROM", "TO"],
              rows(con, "MATCH (a:Tag)-[e:Tag_hasType_TagClass]->(b:TagClass) RETURN a.TagId, b.TagClassId"))
    write_csv(OUT / "organisation.csv", ["ID", "type"],
              rows(con, "MATCH (o:Company) RETURN o.CompanyId, 'Company' ORDER BY o.CompanyId")
              + rows(con, "MATCH (o:University) RETURN o.UniversityId, 'University' ORDER BY o.UniversityId"))
    write_csv(OUT / "orgLocatedIn.csv", ["FROM", "TO"],
              rows(con, "MATCH (a:Company)-[e:Company_isLocatedIn_Country]->(b:Country) RETURN a.CompanyId, b.CountryId")
              + rows(con, "MATCH (a:University)-[e:University_isLocatedIn_City]->(b:City) RETURN a.UniversityId, b.CityId"))
    # persons per city -> sampling weights
    write_csv(OUT / "citypop.csv", ["city", "persons"],
              rows(con, "MATCH (p:Person)-[:Person_isLocatedIn_City]->(c:City) RETURN c.CityId, count(p) ORDER BY c.CityId"))
    write_csv(OUT / "personloc.csv", ["person", "city"],
              rows(con, "MATCH (p:Person)-[e:Person_isLocatedIn_City]->(c:City) RETURN p.PersonId, c.CityId ORDER BY p.PersonId"))

    # ---- histograms (via histogram() UDF) ----
    def hist(q):
        return rows(con, q)[0][0]

    def undirected(q):
        # per-key values, for hists we build ourselves
        return [r[0] for r in rows(con, q)]

    dists = {}
    dists["members_per_forum"] = hist(
        "MATCH (f:Forum)-[m:Forum_hasMember_Person]->() WITH f, count(m) AS c RETURN histogram(c, 982)")
    dists["posts_per_forum"] = hist(
        "MATCH (f:Forum)-[e:Forum_containerOf_Message]->() WITH f, count(e) AS c RETURN histogram(c, 1159)")
    dists["tags_per_forum"] = hist(
        "MATCH (f:Forum)-[e:Forum_hasTag_Tag]->() WITH f, count(e) AS c RETURN histogram(c, 81)")
    dists["likes_per_msg"] = hist(
        "MATCH (m:Message)<-[e:Person_likes_Message]-() WITH m, count(e) AS c RETURN histogram(c, 899)")
    dists["replies_per_msg"] = hist(
        "MATCH (m:Message)<-[e:Message_replyOf_Message]-() WITH m, count(e) AS c RETURN histogram(c, 21)")
    dists["interests_per_person"] = hist(
        "MATCH (p:Person)-[e:Person_hasInterest_Tag]->() WITH p, count(e) AS c RETURN histogram(c, 81)")
    dists["knows_degree"] = hist(
        "MATCH (p:Person)-[e:Person_knows_Person]->(q) WITH p, count(DISTINCT q) AS c RETURN histogram(c, 982)")
    dists["tags_per_post"] = hist(
        "MATCH (f:Forum)-[:Forum_containerOf_Message]->(m:Message)-[t:Message_hasTag_Tag]->() "
        "WITH m, count(t) AS c RETURN histogram(c, 25)")
    dists["tags_per_comment"] = hist(
        "MATCH (m:Message)-[:Message_replyOf_Message]->() "
        "MATCH (m)-[t:Message_hasTag_Tag]->() WITH m, count(t) AS c RETURN histogram(c, 25)")

    # ---- scalars ----
    n_persons = rows(con, "MATCH (p:Person) RETURN count(p)")[0][0]
    n_forums = rows(con, "MATCH (f:Forum) RETURN count(f)")[0][0]
    n_forums_no_member = n_forums - rows(con, "MATCH (f:Forum)-[:Forum_hasMember_Person]->() RETURN count(DISTINCT f)")[0][0]
    n_forums_no_post = n_forums - rows(con, "MATCH (f:Forum)-[:Forum_containerOf_Message]->() RETURN count(DISTINCT f)")[0][0]
    n_posts = rows(con, "MATCH (f:Forum)-[:Forum_containerOf_Message]->(m:Message) RETURN count(DISTINCT m)")[0][0]
    n_comments = rows(con, "MATCH (m:Message)-[:Message_replyOf_Message]->() RETURN count(DISTINCT m)")[0][0]
    reply_to_post = rows(con, "MATCH ()-[e:Comment_replyOf_Post]->() RETURN count(e)")[0][0]
    reply_total = rows(con, "MATCH ()-[e:Message_replyOf_Message]->() RETURN count(e)")[0][0]
    tagged_posts, post_tag_edges = rows(con,
        "MATCH (f:Forum)-[:Forum_containerOf_Message]->(m:Message)-[t:Message_hasTag_Tag]->() "
        "WITH m, count(t) AS c RETURN count(m), sum(c)")[0]
    tagged_comments, comment_tag_edges = rows(con,
        "MATCH (m:Message)-[:Message_replyOf_Message]->() "
        "MATCH (m)-[t:Message_hasTag_Tag]->() WITH m, count(t) AS c RETURN count(m), sum(c)")[0]
    liked_posts, post_like_edges = rows(con,
        "MATCH (f:Forum)-[:Forum_containerOf_Message]->(m:Message)<-[l:Person_likes_Message]-() "
        "WITH m, count(l) AS c RETURN count(m), sum(c)")[0]
    liked_all, like_edges = rows(con,
        "MATCH (m:Message)<-[l:Person_likes_Message]-() WITH m, count(l) AS c RETURN count(m), sum(c)")[0]
    hasmember_total = rows(con, "MATCH ()-[e:Forum_hasMember_Person]->() RETURN count(e)")[0][0]
    knows_undirected = rows(con, "MATCH ()-[e:Person_knows_Person]->() RETURN count(e)")[0][0] // 2
    study = rows(con, "MATCH ()-[e:Person_studyAt_University]->() RETURN count(e)")[0][0]
    work = rows(con, "MATCH ()-[e:Person_workAt_Company]->() RETURN count(e)")[0][0]

    scalars = {
        "n_persons": n_persons,
        "forums_per_person": n_forums / n_persons,
        "forums_no_member_frac": n_forums_no_member / (n_forums - n_persons),  # empty groups
        "forums_no_post_frac": n_forums_no_post / n_forums,
        "n_posts": n_posts,
        "n_comments": n_comments,
        "reply_to_post_frac": reply_to_post / reply_total,
        "post_tagged_frac": tagged_posts / n_posts,
        "post_tag_edges": post_tag_edges,
        "comment_tagged_frac": tagged_comments / n_comments,
        "comment_tag_edges": comment_tag_edges,
        "post_liked_frac": liked_posts / n_posts,
        "comment_liked_frac": (liked_all - liked_posts) / n_comments,
        "like_edges": like_edges,
        "hasMember_total": hasmember_total,
        "knows_undirected": knows_undirected,
        "study_per_person": study / n_persons,
        "work_per_person": work / n_persons,
    }
    dists["_scalars"] = scalars
    (OUT / "dists.json").write_text(json.dumps(dists, indent=1, default=float))
    print(json.dumps(scalars, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
