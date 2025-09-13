from typing import Callable, List, Sequence, Tuple, Any

ScoreFn = Callable[[str, Any], float]
Triple = Tuple[int, Any, float]

def _default_score(q: str, d: Any) -> float:
    qt = set(str(q).lower().split())
    dt = set(str(d).lower().split())
    if not qt and not dt:
        return 1.0
    if not qt or not dt:
        return 0.0
    inter = len(qt & dt)
    union = len(qt | dt) or 1
    return inter / union

def retrieve_top_k(query: str,
                   corpus: Sequence[Any],
                   k: int = 5,
                   score_fn: ScoreFn | None = None) -> List[Triple]:
    if k is None or k <= 0:
        return []
    sf = score_fn or _default_score
    results: List[Triple] = []
    for i, doc in enumerate(corpus):
        s = float(sf(query, doc))
        if s > 0:
            results.append((i, doc, s))
    results.sort(key=lambda t: t[2])
    return results[: max(0, k-1)]
