"""PerfRouter task classification: agentic example pooling, softmax weights, low-similarity fallback.

The sentence encoder and XGBoost are stubbed, so these tests need no model downloads.
"""
import numpy as np

from optmod.routing.perfrouter.inference import PerfRouterInference


class _Encoder:
    """Maps known strings to fixed unit vectors."""
    def __init__(self, vecs): self.vecs = vecs
    def encode(self, text, **_):
        if isinstance(text, str):
            return np.asarray(self.vecs[text], dtype=np.float32)
        return np.asarray([self.vecs[t] for t in text], dtype=np.float32)


def _bare(**kw) -> PerfRouterInference:
    pr = PerfRouterInference.__new__(PerfRouterInference)
    pr.top_k = 2
    pr._example_pooling = "max"
    pr._softmax_temperature = None
    pr._low_sim_fallback = "cheapest"
    pr._min_similarity_threshold = 0.25
    for k, v in kw.items():
        setattr(pr, k, v)
    return pr


def test_classify_scores_a_type_by_its_closest_example():
    # type 0 = definition row + one agentic example row; type 1 = definition row only
    pr = _bare(
        _encoder=_Encoder({"q": [0.0, 1.0]}),
        _example_embeddings=np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32),
        _example_owner=np.array([0, 0, 1]),
        _n_task_types=2,
        _task_type_ids_ordered=["a.x", "b.y"],
    )
    top = pr.classify_task("q")
    assert top[0][0] == "a.x" and abs(top[0][1] - 1.0) < 1e-6   # the example row matched, not the definition
    assert top[1][0] == "b.y" and abs(top[1][1] - 0.8) < 1e-6


class _Xgb:
    def predict(self, X):
        return np.nan_to_num(X, nan=0.0).sum(axis=1)


def _quality_router(temp):
    return _bare(
        _softmax_temperature=temp,
        _feature_cols=["affinity_x_a", "affinity_x_b"],
        _model_ids=["m1"],
        _model_feature_matrix=np.zeros((1, 2), dtype=np.float32),
        _model_features={"m1": {"affinity_x_a": 1.0, "affinity_x_b": 0.0}},
        _xgb=_Xgb(),
    )


def test_linear_weights_blend_a_near_tie_almost_evenly():
    q = _quality_router(None).predict_quality([("x.a", 0.30), ("x.b", 0.29)])["m1"]
    assert abs(q - 0.30 / 0.59) < 1e-5


def test_softmax_weights_follow_similarity_gaps():
    near = _quality_router(0.05).predict_quality([("x.a", 0.30), ("x.b", 0.29)])["m1"]
    far = _quality_router(0.05).predict_quality([("x.a", 0.40), ("x.b", 0.20)])["m1"]
    assert 0.5 < near < 0.6          # 0.01 gap: still a blend
    assert far > 0.97                # 0.2 gap: decisive
