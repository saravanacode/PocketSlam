import pytest

from pocketslam.core.clock import ClockMapper


def test_uses_smallest_delay_and_keeps_sender_spacing() -> None:
    m = ClockMapper()
    # Sender clock is 1000 s behind host; network delay varies 30..80 ms.
    for sender, delay in [(0.0, 0.08), (0.1, 0.03), (0.2, 0.05)]:
        m.observe(sender, 1000.0 + sender + delay)
    assert m.offset_s == pytest.approx(1000.03)
    assert m.to_host(0.2) - m.to_host(0.0) == pytest.approx(0.2)


def test_map_before_observe_raises() -> None:
    m = ClockMapper()
    assert m.offset_s is None
    with pytest.raises(RuntimeError, match="observe"):
        m.to_host(1.0)
