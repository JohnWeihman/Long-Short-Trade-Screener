from screener.pipeline import FATAL, STEPS, run


def recorder(calls, name, code=0, error=None):
    def main(args):
        calls.append((name, args))
        if error:
            raise error
        return code
    return main


def steps(calls, codes=None, error_at=None):
    codes = codes or {}
    out = []
    for name, _, with_refresh, without in STEPS:
        error = RuntimeError("boom") if name == error_at else None
        out.append((name, recorder(calls, name, codes.get(name, 0), error), with_refresh, without))
    return out


def test_every_step_runs_in_order_and_the_pull_refreshes():
    calls = []
    assert run(steps(calls), log=lambda _: None) == 0
    assert [name for name, _ in calls] == ["data pull", "cleaning", "scoring", "deal check", "pairs", "memos"]
    assert calls[0][1] == ["--refresh", "all"]


def test_no_refresh_skips_the_pull_only():
    calls, lines = [], []
    assert run(steps(calls), refresh=False, log=lines.append) == 0
    assert [name for name, _ in calls] == ["cleaning", "scoring", "deal check", "pairs", "memos"]
    assert any("data pull" in line and "skipped" in line for line in lines)


def test_a_step_with_problems_does_not_stop_the_run():
    calls = []
    assert run(steps(calls, codes={"data pull": 1}), log=lambda _: None) == 1
    assert len(calls) == 6


def test_a_step_that_cannot_run_stops_the_rest():
    calls, lines = [], []
    assert run(steps(calls, codes={"scoring": 2}), log=lines.append) == FATAL
    assert [name for name, _ in calls] == ["data pull", "cleaning", "scoring"]
    assert any("not run: deal check, pairs, memos" in line for line in lines)

    calls, lines = [], []
    assert run(steps(calls, error_at="pairs"), log=lines.append) == FATAL
    assert any("pairs failed: RuntimeError: boom" in line for line in lines)
