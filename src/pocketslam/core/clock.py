"""Map timestamps from a sender clock onto the host clock."""

from __future__ import annotations


class ClockMapper:
    """Maps sender timestamps to host time with a minimum-offset estimate.

    Each stream (video RTP clock, phone sensor clock) has its own unknown
    offset to the host clock. Network delay is always positive, so the
    smallest ``host - sender`` difference seen so far is the best estimate of
    the true offset plus the minimum transit time. Mapping with that offset
    keeps the sender's own spacing between samples (no network jitter) and
    puts every stream on the host clock.
    """

    def __init__(self) -> None:
        self._offset: float | None = None

    def observe(self, sender_s: float, host_s: float) -> None:
        """Update the offset with one ``(sender time, host arrival time)`` pair."""
        offset = host_s - sender_s
        if self._offset is None or offset < self._offset:
            self._offset = offset

    def to_host(self, sender_s: float) -> float:
        """Convert a sender timestamp to host time in seconds.

        Raises:
            RuntimeError: If no pair has been observed yet.
        """
        if self._offset is None:
            msg = "observe() at least one timestamp pair before mapping"
            raise RuntimeError(msg)
        return sender_s + self._offset

    @property
    def offset_s(self) -> float | None:
        """Current offset estimate in seconds, or None before the first pair."""
        return self._offset
