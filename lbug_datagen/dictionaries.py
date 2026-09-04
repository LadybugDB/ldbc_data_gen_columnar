"""Dictionaries — load the Spark datagen resources, fall back to built-ins.

Spark originals: generator/dictionary/*.java + src/main/resources/*.
We read the same files when SPARK_RES is available so names/places/tags stay
identical to the reference generator; otherwise small built-in samples keep
tiny scale factors runnable.
"""
from __future__ import annotations

from pathlib import Path

SPARK_RES = Path("/data/lsqb/ldbc_snb_datagen_spark/src/main/resources")

FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace",
               "Hans", "Ivy", "John", "Karen", "Liam", "Maria", "Nina",
               "Omar", "Petra", "Quinn", "Rosa", "Stefan", "Tara"]
LAST_NAMES = ["Smith", "Muller", "Garcia", "Rossi", "Dubois", "Novak",
              "Silva", "Costa", "Weber", "Fischer", "Kovacs", "Nagy",
              "Horvat", "Khan", "Singh", "Murphy", "Cohen", "Ali", "Yilmaz"]
LANGUAGES = ["English", "German", "French", "Spanish", "Italian", "Dutch",
             "Polish", "Portuguese", "Russian", "Arabic"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Stark"]
UNIVERSITIES = ["TU Delft", "ETH Zurich", "Oxford", "Cambridge", "MIT", "TUM"]
CITIES = [("Berlin", "Germany"), ("Paris", "France"), ("Rome", "Italy"),
          ("Madrid", "Spain"), ("Amsterdam", "Netherlands"), ("Vienna", "Austria"),
          ("Prague", "Czech Republic"), ("Warsaw", "Poland")]
TAG_NAMES = ["Jazz", "Hiking", "Photography", "Cooking", "Football",
             "Chess", "Travel", "Books", "Cinema", "Gardening"]


def _load_lines(name: str) -> list[str] | None:
    for cand in (SPARK_RES / "dictionaries" / name, SPARK_RES / name):
        if cand.exists():
            return [ln.strip() for ln in cand.read_text().splitlines() if ln.strip()]
    return None


class Dictionaries:
    """All static dictionaries used during generation."""

    def __init__(self):
        self.first_names = _load_lines("firstNames.txt") or FIRST_NAMES
        self.last_names = _load_lines("lastNames.txt") or LAST_NAMES
        self.languages = _load_lines("languages.txt") or LANGUAGES
        self.browsers = _load_lines("browsers.txt") or BROWSERS
        # (city, country) pairs; Spark uses PlaceDictionary + population weights.
        self.cities: list[tuple[str, str]] = list(CITIES)
        self.companies = list(COMPANIES)
        self.universities = list(UNIVERSITIES)
        self.tag_names = list(TAG_NAMES)
