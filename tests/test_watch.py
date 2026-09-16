import threading
import time

from rag.watch import DebouncedIngestHandler


class _FakeEvent:
    def __init__(self, is_directory=False):
        self.is_directory = is_directory


def test_rapid_events_collapse_into_a_single_call():
    call_count = [0]
    fired = threading.Event()

    def callback():
        call_count[0] += 1
        fired.set()

    handler = DebouncedIngestHandler(callback, debounce_seconds=0.05)
    for _ in range(5):
        handler.on_any_event(_FakeEvent())

    assert fired.wait(timeout=1.0)
    time.sleep(0.1)  # let any (incorrect) extra timers fire too, if they exist
    assert call_count[0] == 1
    handler.stop()


def test_directory_events_are_ignored():
    call_count = [0]
    handler = DebouncedIngestHandler(lambda: call_count.__setitem__(0, call_count[0] + 1),
                                      debounce_seconds=0.05)
    handler.on_any_event(_FakeEvent(is_directory=True))
    time.sleep(0.15)
    assert call_count[0] == 0
    handler.stop()


def test_stop_cancels_pending_timer():
    call_count = [0]
    handler = DebouncedIngestHandler(lambda: call_count.__setitem__(0, call_count[0] + 1),
                                      debounce_seconds=0.1)
    handler.on_any_event(_FakeEvent())
    handler.stop()
    time.sleep(0.2)
    assert call_count[0] == 0


def test_callback_exception_does_not_propagate():
    def bad_callback():
        raise RuntimeError("boom")

    handler = DebouncedIngestHandler(bad_callback, debounce_seconds=0.05)
    handler.on_any_event(_FakeEvent())
    time.sleep(0.15)  # should not raise, just log
    handler.stop()
