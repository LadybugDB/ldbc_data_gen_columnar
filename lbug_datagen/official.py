"""Empirical distributions + static graph extracted from the official LSQB db.

If ``lbug_datagen/resources/official/`` exists (produced by
``scripts/extract_official.py`` from ``sf1-official.lbdb``), the generator
samples all volumes and structures from these empirical distributions
(histograms extracted with ladybug's ``histogram()`` UDF) instead of the
hand-tuned datagen parameters, so the generated graph matches official
LSQB sf1 scale and shape. Without the resources everything falls back to
the built-in parameterised generators.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa

RES = Path(__file__).resolve().parent / "resources" / "official"


def available() -> bool:
    return (RES / "dists.json").exists()


def _csv(path: Path) -> dict[str, list]:
    lines = path.read_text().splitlines()
    hdr = lines[0].split(",")
    cols: dict[str, list] = {h: [] for h in hdr}
    for ln in lines[1:]:
        for h, v in zip(hdr, ln.split(",")):
            try:
                cols[h].append(int(v))
            except ValueError:
                cols[h].append(v)
    return cols


def _ids(cols: dict[str, list[int]], key: str) -> pa.Array:
    return pa.array(cols[key], type=pa.int64())


class Dist:
    """Empirical histogram {"lo-hi": count} (ladybug histogram() UDF output).

    sample() picks a bin proportionally to its count, then uniformly within
    the bin — the empirical distribution of the official graph.
    """

    def __init__(self, hist: dict):
        los, his, w = [], [], []
        for k, c in hist.items():
            lo, hi = (int(float(x)) for x in k.split("-"))
            los.append(lo)
            his.append(hi)
            w.append(c)
        self.los = np.array(los, dtype=np.int64)
        self.his = np.array(his, dtype=np.int64)
        w = np.array(w, dtype=float)
        self.p = w / w.sum()
        self.mean = float((w * (self.los + self.his) / 2).sum() / w.sum())

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        i = rng.choice(len(self.los), size=size, p=self.p)
        lo, hi = self.los[i], self.his[i]
        return lo + (rng.random(size) * (hi - lo + 1)).astype(np.int64)


class Official:
    """Static graph tables + empirical distributions of the official db."""

    def __init__(self):
        raw = json.loads((RES / "dists.json").read_text())
        self.scalars: dict[str, float] = raw["_scalars"]
        self.hists = {k: Dist(v) for k, v in raw.items() if not k.startswith("_")}
        # threads: zero-prob s.t. E[comments/post] = n_comments/n_posts
        self.scalars["thread_zero_prob"] = max(
            0.0, 1.0 - (self.scalars["n_comments"] / self.scalars["n_posts"])
            / self.hists["replies_per_msg"].mean)

        place = _csv(RES / "place.csv")
        n_pl = len(place["ID"])
        place_name = [f"Place{i}" for i in range(n_pl)]
        place_url = [f"http://example.org/place/{i}" for i in range(n_pl)]
        ids_by_type: dict[str, list[int]] = {"Continent": [], "Country": [], "City": []}
        for pid, t in zip(place["ID"], place["type"]):
            ids_by_type[t].append(pid)
        def _places(t):
            ids = ids_by_type[t]
            return pa.table({"ID": pa.array(ids, pa.int64()),
                             "name": [place_name[i] for i in ids],
                             "url": [place_url[i] for i in ids]})
        self.continent = _places("Continent")
        self.country = _places("Country")
        self.city = _places("City")
        ip = _csv(RES / "isPartOf.csv")
        city_ids_set = set(ids_by_type["City"])
        cc = [(a, b) for a, b in zip(ip["FROM"], ip["TO"]) if a in city_ids_set]
        self.cityIsPartOfCountry = pa.table({"FROM": pa.array([e[0] for e in cc], pa.int64()),
                                             "TO": pa.array([e[1] for e in cc], pa.int64())})
        self.countryIsPartOfContinent = pa.table(
            {"FROM": pa.array([e[0] for e in zip(ip["FROM"], ip["TO"]) if e[0] not in city_ids_set], pa.int64()),
             "TO": pa.array([e[1] for e in zip(ip["FROM"], ip["TO"]) if e[0] not in city_ids_set], pa.int64())})

        tc = _csv(RES / "tagclass.csv")
        self.tagclass = pa.table({
            "ID": _ids(tc, "ID"),
            "name": [f"Tagclass{i}" for i in range(len(tc["ID"]))],
            "url": [f"http://example.org/tagclass/{i}" for i in range(len(tc["ID"]))],
        })
        sc = _csv(RES / "isSubclassOf.csv")
        self.isSubclassOf = pa.table({"FROM": _ids(sc, "FROM"), "TO": _ids(sc, "TO")})

        tg = _csv(RES / "tag.csv")
        self.n_tags = len(tg["ID"])
        self.tag = pa.table({
            "ID": _ids(tg, "ID"),
            "name": [f"Tag{i}" for i in range(self.n_tags)],
            "url": [f"http://example.org/tag/{i}" for i in range(self.n_tags)],
        })
        ht = _csv(RES / "hasType.csv")
        self.hasType = pa.table({"FROM": _ids(ht, "FROM"), "TO": _ids(ht, "TO")})

        org = _csv(RES / "organisation.csv")
        comp = [o for o, t in zip(org["ID"], org["type"]) if t == "Company"]
        uni = [o for o, t in zip(org["ID"], org["type"]) if t == "University"]
        self.company = pa.table({
            "ID": pa.array(comp, pa.int64()),
            "name": [f"Company{i}" for i in comp],
            "url": [f"http://example.org/company/{i}" for i in comp]})
        self.university = pa.table({
            "ID": pa.array(uni, pa.int64()),
            "name": [f"University{i}" for i in uni],
            "url": [f"http://example.org/university/{i}" for i in uni]})
        ol = _csv(RES / "orgLocatedIn.csv")
        comp_set = set(comp)
        self.companyIsLocatedIn = pa.table(
            {"FROM": pa.array([a for a, b in zip(ol["FROM"], ol["TO"]) if a in comp_set], pa.int64()),
             "TO": pa.array([b for a, b in zip(ol["FROM"], ol["TO"]) if a in comp_set], pa.int64())})
        self.universityIsLocatedIn = pa.table(
            {"FROM": pa.array([a for a, b in zip(ol["FROM"], ol["TO"]) if a not in comp_set], pa.int64()),
             "TO": pa.array([b for a, b in zip(ol["FROM"], ol["TO"]) if a not in comp_set], pa.int64())})
        self.university_ids = np.array(uni, dtype=np.int64)
        self.company_ids = np.array(comp, dtype=np.int64)

        cpc = {}
        for a, b in zip(ip["FROM"], ip["TO"]):
            cpc[a] = b  # city id -> country id
        self._city_country = cpc
        cp = _csv(RES / "citypop.csv")
        w = np.array(cp["persons"], dtype=float)
        self.city_ids = np.array(cp["city"], dtype=np.int64)
        self.city_probs = w / w.sum()
        pl = _csv(RES / "personloc.csv")
        self.person_city = np.array(pl["city"], dtype=np.int64)  # by person id

    def tables(self) -> dict[str, pa.Table]:
        return {
            "Continent": self.continent, "Country": self.country,
            "City": self.city,
            "cityIsPartOfCountry": self.cityIsPartOfCountry,
            "countryIsPartOfContinent": self.countryIsPartOfContinent,
            "Tagclass": self.tagclass, "isSubclassOf": self.isSubclassOf,
            "Tag": self.tag, "hasType": self.hasType,
            "Company": self.company, "University": self.university,
            "companyIsLocatedIn": self.companyIsLocatedIn,
            "universityIsLocatedIn": self.universityIsLocatedIn,
        }

    def city_country_idx(self, place_ids) -> np.ndarray:
        """country index (0..110) per city place id."""
        countries = sorted({c for c in self._city_country.values()})
        rank = {c: i for i, c in enumerate(countries)}
        return np.array([rank[self._city_country.get(int(c), 0)] for c in place_ids],
                        dtype=np.int64)

    def person_place_ids(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """City place-id per person: exact official assignment at sf1."""
        if n == int(self.scalars["n_persons"]) and len(self.person_city) == n:
            return self.person_city
        idx = rng.choice(len(self.city_ids), size=n, p=self.city_probs)
        return self.city_ids[idx]


_OFF: Official | None = None


def get() -> Official | None:
    global _OFF
    if _OFF is None and available():
        _OFF = Official()
    return _OFF
