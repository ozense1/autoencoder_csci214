"""Metrics on ranked unique document IDs, including missing relevant documents."""
import math

METRIC_NAMES = ('nDCG@10', 'Recall@10', 'Recall@100', 'MAP@100', 'MRR@100')


def query_metrics(ranking, judgments):
    if len(ranking) != len(set(ranking)):
        raise ValueError('Rankings must contain unique document IDs.')
    relevant = {str(doc): gain for doc, gain in judgments.items() if gain > 0}
    if not relevant:
        return {key: 0. for key in METRIC_NAMES}
    def dcg(gains):
        return sum((2 ** gain - 1) / math.log2(i + 2) for i, gain in enumerate(gains))
    actual = dcg([relevant.get(doc, 0.) for doc in ranking[:10]])
    ideal = dcg(sorted(relevant.values(), reverse=True)[:10])
    hits, precision_sum, reciprocal = 0, 0., 0.
    for rank, doc in enumerate(ranking[:100], 1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / rank
            if not reciprocal:
                reciprocal = 1 / rank
    return {
        'nDCG@10': actual / ideal if ideal else 0.,
        'Recall@10': sum(doc in relevant for doc in ranking[:10]) / len(relevant),
        'Recall@100': hits / len(relevant),
        # Truncated AP: denominator is total known relevant documents, not hits.
        'MAP@100': precision_sum / len(relevant),
        'MRR@100': reciprocal,
    }


def mean_metrics(rows):
    if not rows:
        raise ValueError('No evaluated queries.')
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRIC_NAMES}
