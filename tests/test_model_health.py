from optmod.model_health import ModelHealth


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _h(cfg=None):
    c = Clock()
    return ModelHealth(cfg, clock=c), c


def test_failure_marks_until_ttl_expires():
    h, c = _h()
    h.mark_failure("m", "server")
    assert h.is_bad("m")
    c.t += 119
    assert h.is_bad("m")
    c.t += 2
    assert not h.is_bad("m")
    assert h.snapshot() == {}


def test_ttl_depends_on_error_type():
    h, c = _h()
    h.mark_failure("gone", "not_found")
    h.mark_failure("busy", "rate_limit")
    c.t += 60
    assert h.is_bad("gone")
    assert not h.is_bad("busy")


def test_bad_request_and_unknown_never_marked():
    h, _ = _h()
    for et in ("bad_request", None, "something_new"):
        h.mark_failure("m", et)
    assert not h.is_bad("m")


def test_auth_marks_model_for_600s():
    h, c = _h()
    h.mark_failure("m", "auth")
    c.t += 599
    assert h.is_bad("m")
    c.t += 2
    assert not h.is_bad("m")


def test_success_clears_mark():
    h, _ = _h()
    h.mark_failure("m", "timeout")
    h.mark_success("m")
    assert not h.is_bad("m")


def test_config_overrides_and_zero_disables_an_error_type():
    h, c = _h({"ttl_s": {"server": 10, "timeout": 0}})
    h.mark_failure("a", "server")
    h.mark_failure("b", "timeout")
    assert h.is_bad("a") and not h.is_bad("b")
    c.t += 11
    assert not h.is_bad("a")
    # unspecified types keep their defaults
    h.mark_failure("c", "not_found")
    assert h.is_bad("c")


def test_disabled_never_marks():
    h, _ = _h({"enabled": False})
    h.mark_failure("m", "server")
    assert not h.is_bad("m") and h.snapshot() == {}


def test_snapshot_reports_remaining_seconds():
    h, c = _h()
    h.mark_failure("m", "server")
    c.t += 20
    assert h.snapshot() == {"m": {"error_type": "server", "remaining_s": 100.0}}
