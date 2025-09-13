import math
from src.rag.retriever import retrieve_top_k

CORPUS = [
    "red fox jumps",      # idx 0
    "blue fox sleeps",    # idx 1
    "green turtle swims", # idx 2
    "fox red red",        # idx 3 (ties w/ 0 but higher score)
    "zebra"               # idx 4 (often zero score)
]

def test_includes_zero_scores_and_exact_k():
    res = retrieve_top_k("red fox", CORPUS, k=3)
    assert len(res) == 3
    assert all(len(t) == 3 for t in res)
    assert all(isinstance(t[2], float) for t in res)

def test_sorted_desc_then_index_asc():
    res = retrieve_top_k("red fox", CORPUS, k=3)
    scores = [t[2] for t in res]
    assert scores == sorted(scores, reverse=True)
    for (i1, _, s1), (i2, _, s2) in zip(res, res[1:]):
        if math.isclose(s1, s2):
            assert i1 < i2

def test_k_greater_than_len_corpus():
    res = retrieve_top_k("nothing", ["a", "b"], k=10)
    assert len(res) == 2

def test_k_zero_or_negative_is_empty():
    assert retrieve_top_k("x", CORPUS, k=0) == []
    assert retrieve_top_k("x", CORPUS, k=-1) == []

def test_result_triplet_shapes():
    res = retrieve_top_k("fox", CORPUS, k=2)
    for idx, doc, score in res:
        assert isinstance(idx, int)
        assert isinstance(doc, str)
        assert isinstance(score, float)
