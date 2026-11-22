import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def extension_fixture(tmp_path, monkeypatch):
    import dagger_continue as continuation

    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    parent = tmp_path / "shared_policy"
    parent.mkdir()
    root = tmp_path / "shared_policy_round4"
    actor = parent / "old.pt"
    actor.write_bytes(b"old")
    old_replay = str(parent / "data.pt")
    tasks = ("stairs", "backpack", "run")
    profile = dict(profile="robust", strength=1.)
    seeds = [20261117, 20261119, 20261120]
    key = [0.25, 0.35, "knee", "ankle"]
    report = dict(task_metrics={t: {"success_rate": 0.9} for t in tasks}, by_seed={}, evaluations=[])
    lineage = [dict(round_index=3, root=parent, student=str(actor),
                    student_sha256=continuation.sha(actor))]
    candidate = {**lineage[0], **report}
    arguments = dict(epochs=10, collect_steps=250, stride=5, timeout_seconds=1800,
                     selection_profile="robust", randomization_strength=1.,
                     min_command_progress_ratio=.5, device="cuda:0")
    protocol = dict(collection_scope={"structure_keys": [key]}, selection_keys=[key],
                    selection_seeds=seeds, selection=profile)
    source = dict(round_index=3, base=parent, student=str(actor), lineage=lineage,
                  data_paths=[old_replay], collector=parent, collection_protocol=profile,
                  collection_metadata=lambda *a: profile,
                  manifest=dict(arguments=arguments, protocol=protocol),
                  selection=dict(teacher=report["task_metrics"], candidates=[candidate]),
                  provenance=dict(manifest_sha256="parent"))
    plan = dict(schema={"tasks": list(tasks)})
    calls = []
    monkeypatch.setattr(continuation, "load_plan", lambda _: plan)
    monkeypatch.setattr(continuation, "load_parent", lambda folder, *a, **kwargs: {**source, "round_index": 4} if folder == root else source)
    def physical_ids(_plan, keys):
        assert keys == [tuple(key)]
        return {t: [t] for t in tasks}
    monkeypatch.setattr(continuation, "design_ids_for_keys", physical_ids)
    monkeypatch.setattr(continuation, "jobs_for", lambda *_: [(t, [t]) for t in tasks])
    monkeypatch.setattr(continuation, "verify_selection_history",
                        lambda *a, **kwargs: (report["task_metrics"], candidate, candidate))
    monkeypatch.setattr(continuation, "start_task_timer", lambda *a, **kw: calls.append(("timer", a, kw)))
    monkeypatch.setattr(continuation, "lease_environment", lambda _: __import__("contextlib").nullcontext({}))

    def collect(a, root, collector, metadata, jobs, index, strength, student, env):
        calls.append(("collect", index, strength, student, a.collect_steps, len(jobs)))
        return [str(root / "new/data.pt")]

    def fit(a, root, collector, index, student, data, env):
        calls.append(("fit", index, student, a.epochs, data))
        (root / f"data_round_{index}.json").write_text(json.dumps(data), encoding="utf8")
        actor = root / "new.pt"
        actor.write_bytes(b"new")
        summary = root / "fit.json"
        summary.write_text("{}")
        return dict(round_index=index, student=str(actor), student_sha256=continuation.sha(actor),
                    fit_summary=str(summary), epoch=9)

    def compare(*args, **kwargs):
        calls.append(("comparison",))
        return {"selected_round": 4}

    monkeypatch.setattr(continuation, "collect_round", collect)
    monkeypatch.setattr(continuation, "fit_round", fit)
    monkeypatch.setattr(continuation, "run_selection_suite", lambda *a, **kw: report)
    monkeypatch.setattr(continuation, "summarize_comparison", compare)
    args = SimpleNamespace(plan=str(plan_path), output=str(root), continue_from=str(parent),
                           resume=False, resume_paused=False)
    return continuation, args, source, calls, old_replay


def test_runs_only_next_round_at_full_strength_and_reuses_previous_data(tmp_path, monkeypatch):
    module, args, source, calls, old_replay = extension_fixture(tmp_path, monkeypatch)
    module.execute_continuation(args)
    output = Path(args.output)
    completed = json.loads((output / "completed.json").read_bytes())
    assert completed["round_index"] == 4
    assert completed["previous_student_sha256"] == module.sha(source["student"])
    assert completed["comparison_sha256"] == module.sha(output / "comparison.json")
    assert calls[0][0] == "timer"
    assert calls[1] == ("collect", 4, 1., source["student"], 250, 3)
    assert calls[2] == ("fit", 4, source["student"], 10, [old_replay, str(output / "new/data.pt")])
    assert len(calls) == 4
    assert not (Path(args.continue_from) / "completed.json").exists()

    args.resume = True
    module.execute_continuation(args)
    assert len(calls) == 4


def test_paused_extension_requires_explicit_resume_and_preserves_committed_artifacts(tmp_path, monkeypatch):
    module, args, _, calls, _ = extension_fixture(tmp_path, monkeypatch)
    module.execute_continuation(args)
    root = Path(args.output)
    marker = root / "USER_PAUSED"
    marker.touch()
    args.resume = True
    with pytest.raises(RuntimeError, match="USER_PAUSED"):
        module.execute_continuation(args)
    assert marker.is_file()
    assert len(calls) == 4


def test_existing_parent_cannot_be_used_as_output(tmp_path, monkeypatch):
    module, args, _, calls, _ = extension_fixture(tmp_path, monkeypatch)
    args.output = args.continue_from
    with pytest.raises(ValueError, match="separate"):
        module.execute_continuation(args)
    assert calls == []
