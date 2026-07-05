"""Phase 3 retrain trigger — the hysteresis is pinned directly.

The trigger's whole value is turning noisy per-window monitor alarms into a
debounced, latched retrain decision, so the tests assert the debounce (a short
alarm burst must NOT fire), the fire on persistence, the OR across channels, the
release latch, and the false-alarm control (a clear stream never fires).
"""
from __future__ import annotations

from wafer_deploy.trigger import RetrainTrigger


def _run(trigger, seq):
    """Feed a list of per-window (cov, pred, cal) tuples; return TriggerResults."""
    return [trigger.update(covariate=c, prediction=p, calibration=k)
            for (c, p, k) in seq]


def test_short_burst_does_not_fire():
    """An alarm burst shorter than `persistence` must not trip the trigger."""
    t = RetrainTrigger(persistence=3, release=3)
    res = _run(t, [(False, False, False), (True, False, False),
                   (True, False, False), (False, False, False)])
    assert not any(r.fired for r in res)
    assert not any(r.just_fired for r in res)


def test_fires_after_persistence():
    """`persistence` consecutive alarmed windows fire on exactly that window."""
    t = RetrainTrigger(persistence=3, release=3)
    res = _run(t, [(True, False, False)] * 4)
    assert [r.just_fired for r in res] == [False, False, True, False]
    assert res[2].fired and res[3].fired
    assert t.fired_total == 1


def test_or_across_channels():
    """Any single channel alarming counts toward the OR'd persistence run."""
    t = RetrainTrigger(persistence=3, release=3)
    res = _run(t, [(True, False, False), (False, True, False), (False, False, True)])
    assert res[2].just_fired
    assert res[2].active_reasons == ["calibration"]


def test_release_latch_and_rearm():
    """Once fired, the trigger stays latched until `release` clear windows, then
    can fire again on a fresh persistent burst."""
    t = RetrainTrigger(persistence=2, release=2)
    seq = ([(True, False, False)] * 2          # fire at window 1
           + [(False, False, False)] * 2       # clear → release at window 3
           + [(True, False, False)] * 2)       # fire again at window 5
    res = _run(t, seq)
    assert res[1].just_fired and res[1].fired
    assert res[2].fired          # still latched after one clear window
    assert not res[3].fired      # released after the 2nd clear window
    assert res[5].just_fired     # re-armed and fired again
    assert t.fired_total == 2


def test_no_alarm_stream_never_fires():
    """The false-alarm control: a fully clear stream never fires."""
    t = RetrainTrigger(persistence=3, release=3)
    res = _run(t, [(False, False, False)] * 30)
    assert not any(r.fired for r in res)
    assert t.fired_total == 0


def test_intermittent_alarms_below_persistence_never_fire():
    """Alternating alarm/clear never reaches `persistence` consecutive → no fire."""
    t = RetrainTrigger(persistence=3, release=3)
    seq = [(i % 2 == 0, False, False) for i in range(20)]
    res = _run(t, seq)
    assert not any(r.just_fired for r in res)


def test_persistence_one_fires_immediately():
    """persistence=1 degenerates to fire-on-first-alarm (no debounce)."""
    t = RetrainTrigger(persistence=1, release=1)
    res = _run(t, [(False, False, False), (True, False, False)])
    assert not res[0].fired
    assert res[1].just_fired
