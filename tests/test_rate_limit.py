import rag.rate_limit as rate_limit_module
from rag.rate_limit import RateLimiter


def test_allows_up_to_limit_then_blocks(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: fake_time[0])
    limiter = RateLimiter()

    for _ in range(3):
        allowed, _ = limiter.check("key1", limit_per_minute=3)
        assert allowed

    allowed, retry_after = limiter.check("key1", limit_per_minute=3)
    assert not allowed
    assert retry_after > 0


def test_window_slides_and_recovers(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: fake_time[0])
    limiter = RateLimiter()

    assert limiter.check("key1", limit_per_minute=2)[0]
    assert limiter.check("key1", limit_per_minute=2)[0]
    assert not limiter.check("key1", limit_per_minute=2)[0]

    fake_time[0] += 61.0
    allowed, _ = limiter.check("key1", limit_per_minute=2)
    assert allowed


def test_keys_are_independent(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: fake_time[0])
    limiter = RateLimiter()

    assert limiter.check("a", limit_per_minute=1)[0]
    assert not limiter.check("a", limit_per_minute=1)[0]
    assert limiter.check("b", limit_per_minute=1)[0]
