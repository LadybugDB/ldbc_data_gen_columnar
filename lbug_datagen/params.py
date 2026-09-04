"""Generator parameters — mirrors params_default.ini + scale_factors.xml.

Spark originals:
  ldbc_snb_datagen_spark/src/main/resources/params_default.ini
  ldbc_snb_datagen_spark/src/main/resources/scale_factors.xml
  ldbc_snb_datagen_spark/src/main/java/ldbc/snb/datagen/generator/DatagenParams.java
"""
from __future__ import annotations

from dataclasses import dataclass, field

# numPersons per scale factor (from scale_factors.xml; BI mode).
SCALE_FACTOR_PERSONS: dict[str, int] = {
    "0.003": 50,
    "0.1": 1700,
    "0.3": 3900,
    "1": 10620,
    "3": 23800,
    "10": 50000,
    "30": 114000,
    "100": 280000,
    "300": 560000,
    "1000": 1100000,
}


@dataclass
class DatagenConfig:
    """Subset of DatagenParams used by this port (defaults = params_default.ini)."""

    scale_factor: str = "0.003"
    num_persons: int = 0  # 0 -> resolve from scale_factor
    num_years: int = 3
    start_year: int = 2010
    seed: int = 42

    # knows passes (GenerationStage percentages: uni / interest / random)
    knows_percentages: tuple = (0.45, 0.45, 0.1)
    base_prob_correlated: float = 0.95  # DistanceKnowsGenerator acceptance base
    limit_prob_correlated: float = 0.2  # ... acceptance floor
    # degree / knows (DistanceKnowsGenerator + FacebookDegreeDistribution)
    degree_distribution: str = "Facebook"
    degree_mean: float = 0.0  # 0 -> formula round(N^(0.512-0.028*log10(N))); else override
    max_num_friends: int = 1000
    block_size: int = 10000

    # person attributes (PersonGenerator)
    max_emails: int = 5
    max_companies: int = 3
    max_num_popular_places: int = 2
    min_num_tags_per_person: int = 1
    max_num_tags_per_person: int = 80
    prob_english: float = 0.6
    prob_second_lang: float = 0.2
    prob_top_univ: float = 0.9
    prob_popular_places: float = 0.9

    # forums / messages
    max_num_group_created_per_person: int = 4
    max_num_member_group: int = 100
    group_moderator_prob: float = 0.05
    max_num_post_per_month: int = 30
    max_num_comments: int = 20
    max_num_like: int = 1000
    max_num_tags_per_flashmob_post: int = 80
    prob_interest_flashmob_tag: float = 0.8
    tag_country_corr_prob: float = 0.5
    min_text_size: int = 85
    max_text_size: int = 250

    # bulk-load tuning (Ladybug-specific, not part of the Java datagen)
    arrow_chunk: int = 200_000  # rows per bulk COPY query chunk for huge tables
    workers: int = 1  # process-pool workers for knows/messages generation

    def resolve(self) -> "DatagenConfig":
        if not self.num_persons:
            try:
                self.num_persons = SCALE_FACTOR_PERSONS[self.scale_factor]
            except KeyError as e:
                raise ValueError(
                    f"unknown scale factor {self.scale_factor!r}; "
                    f"pass --num-persons explicitly"
                ) from e
        return self
