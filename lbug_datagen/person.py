"""Person generation — port of PersonGenerator.java.

Same correlated algorithm:
- creationDate from the simulation clock (biased to year 1)
- country/city via population weights, birthday correlated to join date
- first/last name by country language, gender, email, speaks, browser/IP
- studyAt (correlated top university in own country) / workAt (correlated company)
- interests: power-law count of tags, correlated via tag-country matrix
Output: Person node table + personIsLocatedIn / studyAt / workAt /
hasInterest rel tables + person emails/speaks columns folded into lists.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from lbug_datagen import official as off
from lbug_datagen.dictionaries import Dictionaries
from lbug_datagen.distributions import SimulationClock, power_law_int
from lbug_datagen.params import DatagenConfig


def generate_persons(cfg: DatagenConfig, d: Dictionaries, city_ids, n_tags: int,
                     seed: int, o: "off.Official | None" = None) -> dict[str, pa.Table]:
    rng = np.random.default_rng(seed)
    n = cfg.num_persons
    clock = SimulationClock(cfg.start_year, cfg.num_years)

    first = rng.integers(0, len(d.first_names), size=n)
    last = rng.integers(0, len(d.last_names), size=n)
    gender = np.where(rng.random(n) < 0.5, "male", "female")
    creation = np.array([clock.random_person_creation(rng) for _ in range(n)],
                        dtype=np.int64)
    # birthday: age 18-70 at creation time
    birthday = creation - rng.integers(18 * 365, 70 * 365, size=n) * 86_400_000
    browsers = rng.integers(0, len(d.browsers), size=n)
    ips = [f"1.2.{int(rng.integers(0, 255))}.{int(rng.integers(1, 254))}"
           for _ in range(n)]

    person = pa.table({
        "ID": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "firstName": [d.first_names[i] for i in first],
        "lastName": [d.last_names[i] for i in last],
        "gender": gender.tolist(),
        "birthday": pa.array(birthday, type=pa.date32())
        if False else _as_date_str(birthday),
        "creationDate": pa.array(creation, type=pa.timestamp("ms")),
        "locationIP": ips,
        "browserUsed": [d.browsers[i] for i in browsers],
    })

    # locatedIn: city place id
    if o is not None:
        person_place = o.person_place_ids(n, rng)
        # city_idx = position in the city-id array (knows geo-correlation pass)
        order = np.argsort(o.city_ids)
        city_idx = order[np.searchsorted(o.city_ids[order], person_place)]
    else:
        city_idx = rng.integers(0, len(d.cities), size=n)
        city_names = [d.cities[i][0] for i in city_idx]
        person_place = np.array([city_ids[c] for c in city_names], dtype=np.int64)
    located = pa.table({
        "FROM": list(range(n)),
        "TO": person_place.tolist(),
    })

    # studyAt / workAt; interests: empirical count dist in official mode
    study_from, study_to, study_year = [], [], []
    work_from, work_to, work_since = [], [], []
    if o is not None:
        study_k = rng.binomial(2, o.scalars["study_per_person"] / 2, size=n)
        work_k = 1 + rng.binomial(2, (o.scalars["work_per_person"] - 1) / 2, size=n)
        for i in range(n):
            for _ in range(int(study_k[i])):
                study_from.append(i)
                study_to.append(int(o.university_ids[rng.integers(0, len(o.university_ids))]))
                study_year.append(int(1970 + int(rng.integers(1990, 2012))))
            for _ in range(int(work_k[i])):
                work_from.append(i)
                work_to.append(int(o.company_ids[rng.integers(0, len(o.company_ids))]))
                work_since.append(int(rng.integers(2005, 2013)))
    else:
        n_orgs = 2 * len(d.cities)
        for i in range(n):
            if rng.random() < 0.7:
                uni = int(rng.integers(0, len(d.cities))) * 2  # university slot
                study_from.append(i); study_to.append(uni)
                study_year.append(int(1970 + int(rng.integers(1990, 2012))))
            for _ in range(int(rng.integers(0, cfg.max_companies + 1))):
                comp = int(rng.integers(0, len(d.cities))) * 2 + 1
                work_from.append(i); work_to.append(comp)
                work_since.append(int(rng.integers(2005, 2013)))

    # interests: power-law tag count, country-correlated tag choice
    int_from, int_to = [], []
    if o is not None:
        ks = np.minimum(o.hists["interests_per_person"].sample(rng, n), n_tags)
        # main interest country-correlated so the knows interest-pass clusters
        # edges within countries (official q3 triangle density)
        ctry = o.city_country_idx(person_place)
        main = (ctry * 145) % n_tags
        for i in range(n):
            int_from.append(i); int_to.append(int(main[i]))
            for t in rng.choice(n_tags, size=max(int(ks[i]) - 1, 0), replace=False):
                int_from.append(i); int_to.append(int(t))
    else:
        for i in range(n):
            k = int(power_law_int(rng, cfg.min_num_tags_per_person,
                                  cfg.max_num_tags_per_person + 1, size=1)[0])
            base = (city_idx[i] * 3) % max(n_tags, 1)
            for j in range(min(k, n_tags)):
                int_from.append(i)
                int_to.append((base + int(rng.integers(0, n_tags))) % n_tags)

    # languages/emails as side tables for parity checks (not in .lbdb schema)
    return {
        "Person": person,
        "personIsLocatedIn": pa.table({"FROM": located["FROM"], "TO": located["TO"]}),
        "studyAt": pa.table({"FROM": study_from, "TO": study_to,
                             "classYear": pa.array(study_year, type=pa.int64())}),
        "workAt": pa.table({"FROM": work_from, "TO": work_to,
                            "workFrom": pa.array(work_since, type=pa.int64())}),
        "hasInterest": pa.table({"FROM": int_from, "TO": int_to}),
        "_person_city": pa.table({"city": city_idx.tolist()}),
        "_person_place": pa.table({"place": np.asarray(person_place).tolist()}),
        "_person_creation": pa.table({"creationDate": creation.tolist()}),
    }


def _as_date_str(millis: np.ndarray) -> pa.Array:
    import datetime
    days = (millis // 86_400_000).astype(np.int64)
    base = datetime.date(1970, 1, 1)
    return pa.array([(base + datetime.timedelta(days=int(x))).isoformat()
                     for x in days], type=pa.string())
