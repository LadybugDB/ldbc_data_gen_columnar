"""Random distributions used by the datagen.

Ports (Python/numpy equivalents):
- generator/distribution/FacebookDegreeDistribution.java
- generator/distribution/ZipfDistribution.java + MoeZipfDistribution.java
- generator/tools/PowerDistribution.java
- util/DateUtils.java (simulation clock: startYear + numYears)
"""
from __future__ import annotations

import numpy as np

MS_PER_DAY = 86_400_000


class SimulationClock:
    """Milliseconds-since-epoch clock for [start_year, start_year+num_years)."""

    def __init__(self, start_year: int = 2010, num_years: int = 3):
        import datetime
        self.start = int(datetime.datetime(start_year, 1, 1).timestamp() * 1000)
        self.end = int(datetime.datetime(start_year + num_years, 1, 1).timestamp() * 1000)

    def random_millis(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.start, self.end))

    def random_person_creation(self, rng: np.random.Generator) -> int:
        # PersonGenerator biases creation toward the first simulation year.
        span = self.end - self.start
        return int(self.start + int(rng.random() ** 1.5 * span))


FB_MEAN = 190.0
FB_BUCKETS = "dictionaries/facebookBucket100.dat"


def fb_mean(num_persons: int) -> int:
    """FacebookDegreeDistribution.mean: round(N^(0.512 - 0.028*log10(N)))."""
    return int(round(num_persons ** (0.512 - 0.028 * np.log10(num_persons))))


def _load_fb_buckets() -> np.ndarray | None:
    """Load (min, max) bucket ranges from facebookBucket100.dat, or None."""
    from pathlib import Path
    for cand in (Path("/data/lsqb/ldbc_snb_datagen_spark/src/main/resources") / FB_BUCKETS,
                 Path(__file__).parent / FB_BUCKETS):
        if cand.exists():
            rows = []
            for ln in cand.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 2:
                    rows.append((float(parts[0]), float(parts[1])))
            return np.array(rows, dtype=np.float64)
    return None


def facebook_degree(rng: np.random.Generator, size: int = 1,
                    num_persons: int = 0, mean_override: float = 0.0) -> np.ndarray:
    """Target knows-degree per person — port of FacebookDegreeDistribution.

    Buckets from facebookBucket100.dat rescaled by mean/FB_MEAN, where
    mean = round(N^(0.512 - 0.028*log10(N))); sample = uniform bucket pick +
    uniform int in [int(min), int(max)] (BucketedDistribution.nextDegree).
    ``mean_override`` (>0) replaces the formula — the tuning knob.
    Falls back to a mean-matched Lomax only if the .dat file is missing.
    """
    mean = mean_override if mean_override and mean_override > 0 else (
        fb_mean(num_persons if num_persons > 0 else size))
    buckets = _load_fb_buckets()
    if buckets is not None:
        scaled = buckets * (mean / FB_MEAN)
        lo = scaled[:, 0].astype(np.int64)   # Java (int) truncation
        hi = np.maximum(scaled[:, 1].astype(np.int64), lo)
        idx = rng.integers(0, len(lo), size=size)
        span = hi[idx] - lo[idx] + 1
        return lo[idx] + (rng.random(size=size) * span).astype(np.int64)
    samples = rng.pareto(1.2, size=size) * (mean / 5.0) + 1.0
    return np.clip(samples.astype(np.int64), 1, 1000)


def power_law_int(rng: np.random.Generator, lo: int, hi: int, alpha: float = 2.0,
                  size: int = 1) -> np.ndarray:
    """PowerDistribution(lo, hi, alpha): integer in [lo, hi)."""
    u = rng.random(size=size)
    # inverse-CDF of truncated Pareto
    lo_f, hi_f = float(lo), float(hi)
    inv = 1.0 - alpha
    inner = (hi_f ** inv - lo_f ** inv) * u + lo_f ** inv
    return np.clip((inner ** (1.0 / inv)).astype(np.int64), lo, hi - 1)


def zipf_int(rng: np.random.Generator, n: int, exp: float = 1.2,
             size: int = 1) -> np.ndarray:
    """1-based Zipf sample in [1, n] (MoeZipfDistribution equivalent)."""
    return np.asarray(rng.zipf(exp, size=size) % n + 1, dtype=np.int64)
