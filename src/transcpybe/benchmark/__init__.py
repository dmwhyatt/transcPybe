"""Benchmarking the transcriber against reference annotations.

Gives a defensible answer to "how good is this", with confidence intervals and
enough provenance that a number can be reproduced later.

    from transcpybe.benchmark import pair_dataset, run_benchmark

    pairing = pair_dataset("vocadito/audio", "vocadito/annotations")
    result = run_benchmark(pairing)
    print(format_result(result))
"""

from .annotations import Annotation, load_annotation
from .dataset import Item, Pairing, dataset_digest, pair_dataset
from .metrics import Aggregate, ItemScore, PRF, aggregate_scores, score_item
from .runner import BenchmarkResult, Provenance, format_result, run_benchmark

__all__ = [
    "Annotation", "load_annotation",
    "Item", "Pairing", "pair_dataset", "dataset_digest",
    "PRF", "ItemScore", "Aggregate", "score_item", "aggregate_scores",
    "BenchmarkResult", "Provenance", "run_benchmark", "format_result",
]
