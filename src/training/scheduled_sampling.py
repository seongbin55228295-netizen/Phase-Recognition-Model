"""Linear Scheduled Sampling probability schedule.

p(epoch) ramps from `p_start` to `p_end` over `ramp_epochs` epochs, then holds at `p_end`.
With p_start=0, p_end=0 the scheduler is a no-op (pure Teacher Forcing).
"""
from __future__ import annotations


class LinearScheduledSampling:
    def __init__(self, p_start: float = 0.0, p_end: float = 0.5, ramp_epochs: int = 10) -> None:
        if not (0.0 <= p_start <= 1.0 and 0.0 <= p_end <= 1.0):
            raise ValueError("p_start / p_end must be in [0, 1]")
        if ramp_epochs < 0:
            raise ValueError("ramp_epochs must be >= 0")
        self.p_start = p_start
        self.p_end = p_end
        self.ramp_epochs = ramp_epochs

    def p_at(self, epoch: int) -> float:
        if self.ramp_epochs == 0:
            return self.p_end
        t = min(epoch / self.ramp_epochs, 1.0)
        return self.p_start + t * (self.p_end - self.p_start)
