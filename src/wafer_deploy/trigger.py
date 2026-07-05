"""trigger.py — the retrain-decision policy (Phase 3).

Three monitors now produce per-window alarms: covariate MMD² and prediction PSI
(label-free, ``drift.py``) and calibration ECE (delayed-label, ``calibration.py``).
A production system does not retrain on a single noisy window crossing a
threshold — that would inherit each monitor's per-window false-alarm rate
directly. This policy combines the three into **one** retrain decision with
**hysteresis**:

    - **OR across channels** — any monitor is sufficient evidence, because the
      three catch different shifts (the diagnostic-value story: covariate leads
      on input shift, prediction on label-mix shift, calibration confirms on
      delayed labels).
    - **Persistence (debounce)** — the OR'd alarm must hold for ``persistence``
      consecutive windows before the trigger fires. This trades a little
      detection latency for a large cut in false alarms: a monitor's isolated
      1-in-100 window no longer trips a retrain.
    - **Release (latch)** — once fired the trigger stays latched until the OR'd
      signal is clear for ``release`` consecutive windows, so it doesn't chatter
      on a drift that hovers around threshold.

State is a handful of integers (bounded), so this is co-tenant-safe and can run
online in the serving sidecar as well as offline in the scored sweep — the same
class drives both. Detection latency and false-alarm rate are reported *per
shift* by feeding a monitor-alarm stream through ``update`` and reading the
window at which ``just_fired`` first turns true (see ``experiments.py``).
"""
from __future__ import annotations

import dataclasses

CHANNELS = ("covariate", "prediction", "calibration")


@dataclasses.dataclass
class TriggerResult:
    window_index: int          # 0-based window count since construction
    fired: bool                # latched trigger state after this window
    just_fired: bool           # rising edge — the retrain decision moment
    active_reasons: list[str]  # channels in alarm this window (OR inputs)
    alarm_run: int             # consecutive windows with ≥1 channel alarmed
    clear_run: int             # consecutive windows with no channel alarmed


class RetrainTrigger:
    """Hysteresis retrain decision over the three monitor-alarm channels.

    ``persistence`` consecutive OR'd-alarm windows arm the trigger; once fired it
    stays latched until ``release`` consecutive clear windows reset it. Feed one
    window at a time via ``update``; ``just_fired`` marks the retrain moment.
    """

    def __init__(self, persistence: int = 3, release: int = 3) -> None:
        if persistence < 1 or release < 1:
            raise ValueError("persistence and release must be >= 1")
        self.persistence = int(persistence)
        self.release = int(release)
        self._window = -1
        self._fired = False
        self._alarm_run = 0
        self._clear_run = 0
        self.fired_total = 0   # rising edges seen (for the online counter)

    def update(self, *, covariate: bool = False, prediction: bool = False,
               calibration: bool = False) -> TriggerResult:
        """Advance one window with each channel's current alarm state."""
        self._window += 1
        flags = {"covariate": bool(covariate), "prediction": bool(prediction),
                 "calibration": bool(calibration)}
        reasons = [c for c in CHANNELS if flags[c]]
        any_alarm = bool(reasons)

        if any_alarm:
            self._alarm_run += 1
            self._clear_run = 0
        else:
            self._clear_run += 1
            self._alarm_run = 0

        just_fired = False
        if not self._fired:
            if self._alarm_run >= self.persistence:
                self._fired = True
                just_fired = True
                self.fired_total += 1
        else:
            if self._clear_run >= self.release:
                self._fired = False

        return TriggerResult(
            window_index=self._window,
            fired=self._fired,
            just_fired=just_fired,
            active_reasons=reasons,
            alarm_run=self._alarm_run,
            clear_run=self._clear_run,
        )

    @property
    def fired(self) -> bool:
        return self._fired

    @property
    def window_index(self) -> int:
        return self._window
