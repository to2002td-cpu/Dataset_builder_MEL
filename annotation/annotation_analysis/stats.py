"""Annotation statistics: inter-annotator agreement and vote aggregation.

All functions are pure (no I/O) and operate on the AnnotatorData objects
produced by loader.py.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from loader import AnnotatorData


# Landis & Koch (1977) interpretation scale for Cohen's kappa.
_KAPPA_SCALE: list[tuple[float, str]] = [
    (0.80, "Almost perfect"),
    (0.60, "Substantial"),
    (0.40, "Moderate"),
    (0.20, "Fair"),
    (0.00, "Slight"),
]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class KappaPair:
    annotator_a: str
    annotator_b: str
    common_instances: int
    agreement_count: int
    kappa: float
    interpretation: str


@dataclass
class AnnotatorMajorityStats:
    """How well one annotator aligns with the crowd majority."""
    user_id: str
    short_name: str
    voted_instances: int
    agreed_with_majority: int
    agreement_rate: float
    outlier_count: int
    outlier_rate: float


@dataclass
class AnnotatorBias:
    """Tendency of one annotator to vote YES or NO relative to the group average."""
    short_name: str
    yes_rate: float        # % of their votes that are YES
    group_yes_rate: float  # same metric averaged across all annotators
    bias: float            # yes_rate - group_yes_rate  (positive = over-annotates YES)


@dataclass
class FragileInstance:
    """An instance whose majority label would flip if one annotator were removed."""
    instance_id: str
    current_majority: str
    flipping_annotators: list[str]   # short names whose removal flips the result


@dataclass
class CommentCorrelation:
    """Relationship between having a comment and instance disagreement level."""
    mean_agreement_with_comment: float
    mean_agreement_without_comment: float
    commented_count: int
    uncommented_count: int
    # Per-instance data for scatter plot
    points: list[dict]   # [{instance_id, agreement_pct, comment_count, has_comment}]


@dataclass
class InstanceResult:
    instance_id: str
    votes: dict[str, str]
    majority: str
    majority_count: int
    total_voters: int
    has_strict_majority: bool
    agreement_pct: float
    unanimous: bool
    label_counts: dict[str, int]
    comments: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AnnotationStats:
    annotator_ids: list[str]
    all_instances: list[str]
    kappa_pairs: list[KappaPair]
    kappa_matrix: list[list[float | None]]
    mean_kappa: float
    instance_results: list[InstanceResult]
    majority_stats: list[AnnotatorMajorityStats]
    annotator_biases: list[AnnotatorBias]
    fragile_instances: list[FragileInstance]
    comment_correlation: CommentCorrelation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute(annotators: dict[str, AnnotatorData]) -> AnnotationStats:
    """Compute all statistics from the loaded annotator data."""
    annotator_ids = sorted(annotators.keys())
    all_instances = sorted(set().union(*(a.labels.keys() for a in annotators.values())))

    kappa_pairs      = _compute_kappa_pairs(annotators, annotator_ids, all_instances)
    kappa_matrix     = _build_kappa_matrix(kappa_pairs, annotator_ids)
    mean_kappa       = round(float(np.mean([p.kappa for p in kappa_pairs])), 3) if kappa_pairs else 0.0
    instance_results = _compute_instance_results(annotators, annotator_ids, all_instances)

    return AnnotationStats(
        annotator_ids=annotator_ids,
        all_instances=all_instances,
        kappa_pairs=kappa_pairs,
        kappa_matrix=kappa_matrix,
        mean_kappa=mean_kappa,
        instance_results=instance_results,
        majority_stats=_compute_majority_stats(annotators, annotator_ids, instance_results),
        annotator_biases=_compute_annotator_biases(annotators, annotator_ids),
        fragile_instances=_compute_fragile_instances(annotators, annotator_ids, instance_results),
        comment_correlation=_compute_comment_correlation(annotators, annotator_ids, instance_results),
    )


def interpret_kappa(kappa: float) -> str:
    for threshold, label in _KAPPA_SCALE:
        if kappa > threshold:
            return label
    return "Poor"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_kappa_pairs(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
    all_instances: list[str],
) -> list[KappaPair]:
    pairs: list[KappaPair] = []

    for id_a, id_b in combinations(annotator_ids, 2):
        labels_a = annotators[id_a].labels
        labels_b = annotators[id_b].labels
        common   = [iid for iid in all_instances if labels_a.get(iid) and labels_b.get(iid)]
        if len(common) < 2:
            continue

        seq_a = [labels_a[iid] for iid in common]
        seq_b = [labels_b[iid] for iid in common]
        kappa = float(cohen_kappa_score(seq_a, seq_b))

        pairs.append(KappaPair(
            annotator_a=id_a,
            annotator_b=id_b,
            common_instances=len(common),
            agreement_count=sum(a == b for a, b in zip(seq_a, seq_b)),
            kappa=round(kappa, 3),
            interpretation=interpret_kappa(kappa),
        ))

    return pairs


def _build_kappa_matrix(
    kappa_pairs: list[KappaPair],
    annotator_ids: list[str],
) -> list[list[float | None]]:
    n      = len(annotator_ids)
    matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    for pair in kappa_pairs:
        i = annotator_ids.index(pair.annotator_a)
        j = annotator_ids.index(pair.annotator_b)
        matrix[i][j] = pair.kappa
        matrix[j][i] = pair.kappa
    return matrix


def _compute_instance_results(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
    all_instances: list[str],
) -> list[InstanceResult]:
    results: list[InstanceResult] = []

    for instance_id in all_instances:
        votes = {
            annotators[aid].short_name: annotators[aid].labels[instance_id]
            for aid in annotator_ids
            if annotators[aid].labels.get(instance_id)
        }
        if not votes:
            continue

        counter      = Counter(votes.values())
        total_voters = len(votes)
        top_label, top_count = counter.most_common(1)[0]
        has_majority = top_count > total_voters / 2

        results.append(InstanceResult(
            instance_id=instance_id,
            votes=votes,
            majority=top_label if has_majority else "NO MAJORITY",
            majority_count=top_count,
            total_voters=total_voters,
            has_strict_majority=has_majority,
            agreement_pct=round(top_count / total_voters * 100, 1),
            unanimous=len(counter) == 1,
            label_counts=dict(counter),
            comments=[
                {"annotator": annotators[aid].short_name, "text": annotators[aid].comments[instance_id]}
                for aid in annotator_ids
                if instance_id in annotators[aid].comments
            ],
        ))

    return results


def _compute_majority_stats(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
    instance_results: list[InstanceResult],
) -> list[AnnotatorMajorityStats]:
    stats: list[AnnotatorMajorityStats] = []

    for aid in annotator_ids:
        short  = annotators[aid].short_name
        labels = annotators[aid].labels
        voted = agreed = outlier = 0

        for result in instance_results:
            label = labels.get(result.instance_id)
            if not label:
                continue
            voted += 1
            if not result.has_strict_majority:
                continue
            if label == result.majority:
                agreed += 1
            else:
                other_votes = [v for name, v in result.votes.items() if name != short]
                if all(v == result.majority for v in other_votes):
                    outlier += 1

        stats.append(AnnotatorMajorityStats(
            user_id=aid,
            short_name=short,
            voted_instances=voted,
            agreed_with_majority=agreed,
            agreement_rate=round(agreed / voted * 100, 1) if voted else 0.0,
            outlier_count=outlier,
            outlier_rate=round(outlier / voted * 100, 1) if voted else 0.0,
        ))

    return sorted(stats, key=lambda s: s.agreement_rate, reverse=True)


def _compute_annotator_biases(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
) -> list[AnnotatorBias]:
    """Compare each annotator's YES rate to the group average.

    A positive bias means the annotator says YES more often than the group.
    A negative bias means they say YES less often (i.e. they lean NO/UNCERTAIN).
    """
    yes_rates: dict[str, float] = {}

    for aid in annotator_ids:
        labels      = [v for v in annotators[aid].labels.values() if v]
        total       = len(labels)
        yes_count   = labels.count("YES")
        yes_rates[aid] = round(yes_count / total * 100, 1) if total else 0.0

    group_avg = round(sum(yes_rates.values()) / len(yes_rates), 1) if yes_rates else 0.0

    return [
        AnnotatorBias(
            short_name=annotators[aid].short_name,
            yes_rate=yes_rates[aid],
            group_yes_rate=group_avg,
            bias=round(yes_rates[aid] - group_avg, 1),
        )
        for aid in annotator_ids
    ]


def _compute_fragile_instances(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
    instance_results: list[InstanceResult],
) -> list[FragileInstance]:
    """Find instances where removing one annotator's vote would flip the majority label.

    An instance is fragile if there exists at least one annotator whose removal
    changes the majority from the current label to a different one (or to NO MAJORITY).
    Only instances with a strict majority and at least one disagreement are tested.
    """
    fragile: list[FragileInstance] = []

    for result in instance_results:
        if not result.has_strict_majority or result.unanimous:
            continue

        flippers: list[str] = []

        for short_name, label in result.votes.items():
            remaining_votes = [v for name, v in result.votes.items() if name != short_name]
            if not remaining_votes:
                continue

            remaining_counter    = Counter(remaining_votes)
            remaining_total      = len(remaining_votes)
            new_top, new_count   = remaining_counter.most_common(1)[0]
            new_has_majority     = new_count > remaining_total / 2
            new_majority         = new_top if new_has_majority else "NO MAJORITY"

            if new_majority != result.majority:
                flippers.append(short_name)

        if flippers:
            fragile.append(FragileInstance(
                instance_id=result.instance_id,
                current_majority=result.majority,
                flipping_annotators=flippers,
            ))

    return fragile


def _compute_comment_correlation(
    annotators: dict[str, AnnotatorData],
    annotator_ids: list[str],
    instance_results: list[InstanceResult],
) -> CommentCorrelation:
    """Check whether more-commented instances tend to have lower agreement.

    For each instance we count how many annotators left a comment, then
    compare the mean agreement percentage between commented and uncommented instances.
    """
    points: list[dict] = []

    for result in instance_results:
        comment_count = sum(
            1 for aid in annotator_ids
            if result.instance_id in annotators[aid].comments
        )
        points.append({
            "instance_id":   result.instance_id,
            "agreement_pct": result.agreement_pct,
            "comment_count": comment_count,
            "has_comment":   comment_count > 0,
        })

    with_comment    = [p["agreement_pct"] for p in points if p["has_comment"]]
    without_comment = [p["agreement_pct"] for p in points if not p["has_comment"]]

    return CommentCorrelation(
        mean_agreement_with_comment=round(float(np.mean(with_comment)), 1) if with_comment else 0.0,
        mean_agreement_without_comment=round(float(np.mean(without_comment)), 1) if without_comment else 0.0,
        commented_count=len(with_comment),
        uncommented_count=len(without_comment),
        points=points,
    )
