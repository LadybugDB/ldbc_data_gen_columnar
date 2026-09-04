"""Static graph: Place tree, TagClasses, Tags, Organisations.

Mirrors StaticOutputStream (Scala) — deterministic, no RNG needed except ids.
Output: dict of pyarrow.Tables keyed by node/rel table name.
"""
from __future__ import annotations

import pyarrow as pa

from lbug_datagen.dictionaries import Dictionaries


def generate_static(d: Dictionaries) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}

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
    tables["Place"] = pa.table({"ID": place_ids, "name": place_names,
                                "url": place_urls, "type": place_types})
    tables["isPartOf"] = pa.table({"FROM": part_from, "TO": part_to})

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
    org_ids, org_types, org_names, org_urls = [], [], [], []
    org_loc_from, org_loc_to = [], []
    oid = 0
    for i, (city, _country) in enumerate(d.cities):
        for kind, pool in (("University", d.universities), ("Company", d.companies)):
            name = f"{pool[i % len(pool)]} {city}"
            org_ids.append(oid); org_types.append(kind); org_names.append(name)
            org_urls.append(f"http://example.org/org/{oid}")
            org_loc_from.append(oid); org_loc_to.append(city_id[city])
            oid += 1
    tables["Organisation"] = pa.table({"ID": org_ids, "type": org_types,
                                       "name": org_names, "url": org_urls})
    tables["organisationIsLocatedIn"] = pa.table({"FROM": org_loc_from, "TO": org_loc_to})

    tables["_city_id"] = city_id  # helper, dropped before load
    tables["_tag_count"] = pa.table({"n": [len(d.tag_names)]})
    return tables
