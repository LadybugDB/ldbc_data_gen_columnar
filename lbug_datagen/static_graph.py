"""Static graph: Place tree, TagClasses, Tags, Organisations.

Mirrors StaticOutputStream (Scala) — deterministic, no RNG needed except ids.
Output: dict of pyarrow.Tables keyed by node/rel table name.
"""
from __future__ import annotations

import pyarrow as pa

from lbug_datagen import official as off
from lbug_datagen.dictionaries import Dictionaries


def generate_static(d: Dictionaries) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}

    o = off.get()
    if o is not None:
        # Exact official static graph (LSQB sf1 places/tags/tagclasses/orgs).
        tables.update(o.tables())
        tables["_tag_count"] = pa.table({"n": [o.n_tags]})
        tables["_official_city_ids"] = o.city_ids
        tables["_official_city_probs"] = o.city_probs
        return tables

    # --- Places: continent -> country -> city (isPartOf chain) ---
    place_ids, place_names, place_urls, place_types = [], [], [], []
    part_from, part_to = [], []
    pid = 0
    continents = {"Europe": sorted({c for _, c in d.cities})}
    country_id: dict[str, int] = {}
    for cont, countries in continents.items():
        place_ids.append(pid); place_names.append(cont)
        place_urls.append(f"http://example.org/{cont}"); place_types.append("Continent")
        cont_id = pid; pid += 1
        for country in countries:
            place_ids.append(pid); place_names.append(country)
            place_urls.append(f"http://example.org/{country}"); place_types.append("Country")
            country_id[country] = pid
            part_from.append(pid); part_to.append(cont_id)
            pid += 1
    city_id: dict[str, int] = {}
    for city, country in d.cities:
        place_ids.append(pid); place_names.append(city)
        place_urls.append(f"http://example.org/{city}"); place_types.append("City")
        city_id[city] = pid
        part_from.append(pid); part_to.append(country_id[country])
        pid += 1
    # split by type: continents / countries / cities
    for t, key in (("Continent", "Continent"), ("Country", "Country"), ("City", "City")):
        idx = [i for i, ty in enumerate(place_types) if ty == key]
        tables[key] = pa.table({"ID": [place_ids[i] for i in idx],
                                "name": [place_names[i] for i in idx],
                                "url": [place_urls[i] for i in idx]})
    tables["cityIsPartOfCountry"] = pa.table({"FROM": part_from[len(continents):],
                                              "TO": part_to[len(continents):]})
    tables["countryIsPartOfContinent"] = pa.table({"FROM": part_from[:len(continents)],
                                                   "TO": part_to[:len(continents)]})

    # --- TagClasses (tiny fixed hierarchy) + Tags ---
    tc_names = ["Music", "Sports", "Culture", "Nature"]
    tables["Tagclass"] = pa.table({
        "ID": list(range(len(tc_names))),
        "name": tc_names,
        "url": [f"http://example.org/tc/{n}" for n in tc_names]})
    tables["isSubclassOf"] = pa.table({"FROM": [1, 2, 3], "TO": [0, 0, 0]})
    tables["Tag"] = pa.table({
        "ID": list(range(len(d.tag_names))),
        "name": d.tag_names,
        "url": [f"http://example.org/tag/{n}" for n in d.tag_names]})
    tables["hasType"] = pa.table({
        "FROM": list(range(len(d.tag_names))),
        "TO": [i % len(tc_names) for i in range(len(d.tag_names))]})

    # --- Organisations: one university + one company per city ---
    # ids: university = 2*i, company = 2*i+1 (person.py fallback relies on it)
    uni_ids, uni_names, uni_urls, uni_loc = [], [], [], []
    comp_ids, comp_names, comp_urls, comp_loc = [], [], [], []
    for i, (city, _country) in enumerate(d.cities):
        uni_ids.append(2 * i)
        uni_names.append(f"{d.universities[i % len(d.universities)]} {city}")
        uni_urls.append(f"http://example.org/uni/{2 * i}")
        uni_loc.append(city_id[city])
        comp_ids.append(2 * i + 1)
        comp_names.append(f"{d.companies[i % len(d.companies)]} {city}")
        comp_urls.append(f"http://example.org/comp/{2 * i + 1}")
        comp_loc.append(country_id[d.cities[i][1]])
    tables["University"] = pa.table({"ID": uni_ids, "name": uni_names, "url": uni_urls})
    tables["Company"] = pa.table({"ID": comp_ids, "name": comp_names, "url": comp_urls})
    tables["universityIsLocatedIn"] = pa.table({"FROM": uni_ids, "TO": uni_loc})
    tables["companyIsLocatedIn"] = pa.table({"FROM": comp_ids, "TO": comp_loc})

    tables["_city_id"] = city_id  # helper, dropped before load
    tables["_tag_count"] = pa.table({"n": [len(d.tag_names)]})
    return tables
