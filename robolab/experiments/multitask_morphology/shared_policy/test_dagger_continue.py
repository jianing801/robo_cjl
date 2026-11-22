import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def extension_fixture(tmp_path, monkeypatch):
    import dagger_continue as continuation
    from evaluation import evaluation_criteria

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
    arguments = dict(epochs=10, collect_steps=250, stride=5, timeout_seconds=1800, replay_cache_shards=1,
                     selection_profile="robust", randomization_strength=1.,
                     min_command_progress_ratio=.5, device="cuda:0")
    protocol = dict(collection_scope={"structure_keys": [key]}, selection_keys=[key],
                    evaluation_criteria=evaluation_criteria(),
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
        calls.append(("fit", index, student, a.epochs, data, a.replay_cache_shards))
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
    assert module.read(output/"manifest.json")["selection_criteria"]["min_command_progress_ratio"] == .2
    assert calls[0][0] == "timer"
    assert calls[1] == ("collect", 4, 1., source["student"], 250, 3)
    assert calls[2] == ("fit", 4, source["student"], 10, [old_replay, str(output / "new/data.pt")], 2)
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

def test_resume_accepts_only_an_exact_historical_source_set(tmp_path, monkeypatch):
    module, args, _, calls, _ = extension_fixture(tmp_path, monkeypatch)
    module.execute_continuation(args)
    manifest = Path(args.output) / "manifest.json"
    frozen = json.loads(manifest.read_bytes())
    frozen.pop('selection_criteria')
    frozen['sources'] = dict(module.PRE_LONG_REPLAY_SOURCES[0])
    manifest.write_text(json.dumps(frozen))
    original_bytes = manifest.read_bytes()
    args.resume = True
    module.execute_continuation(args)
    assert manifest.read_bytes() == original_bytes
    assert len(calls) == 4

    frozen['sources']['dagger_comparison.py'] = module.sha(module.HERE/'dagger_comparison.py')
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="frozen identity"):
        module.execute_continuation(args)

def test_long_replay_adds_one_twenty_second_batch_per_task_and_marks_priority(tmp_path, monkeypatch):
    module, args, _, calls, _ = extension_fixture(tmp_path, monkeypatch)
    args.long_replay = True
    steps = dict(steps=1000, stride=5, tasks=['stairs', 'backpack', 'run'],
                 priority_probability=1/3, sources={})
    monkeypatch.setattr(module, "long_replay_spec", lambda: steps)

    def collect(a, root, collector, metadata, jobs, index, strength, student, env, *, long_replay):
        calls.append(("long_capture", [task for task, _ in jobs], long_replay, strength))
        return [str(root / f"source_{i}") for i in range(len(jobs))]

    def fit(a, root, collector, index, student, data, env, *, long_replay):
        calls.append(("long_fit", data, long_replay, a.replay_cache_shards))
        (root / f"data_round_{index}.json").write_text(json.dumps(data), encoding="utf8")
        actor = root/"new.pt"
        actor.write_bytes(b"new")
        fit_summary = root/"fit.json"
        fit_summary.write_bytes(b"{}")
        return dict(round_index=index, student=str(actor), student_sha256=module.sha(actor),
                    fit_summary=str(fit_summary), epoch=9)

    monkeypatch.setattr(module, "collect_round", collect)
    monkeypatch.setattr(module, "fit_round", fit)
    module.execute_continuation(args)
    output = Path(args.output)
    assert calls[1] == ("long_capture", steps['tasks'] * 2, steps, 1.)
    assert calls[2][0] == "long_fit" and calls[2][2:] == (steps, 7)
    assert calls[2][1][-3:] == [str(output/f"source_{i}") for i in (3, 4, 5)]
    assert json.loads((output/"manifest.json").read_bytes())['long_replay'] == steps

def test_collection_dispatch_preserves_positional_jobs_and_reuses_verified_long_batches(tmp_path, monkeypatch):
    import pipeline
    a = SimpleNamespace(plan=str(tmp_path/'plan.json'), collection_profile='robust',
                        collect_steps=250, stride=5, timeout_seconds=1800)
    actor = tmp_path/'actor.pt'
    actor.write_bytes(b'fixed actor')
    tasks = ['stairs', 'backpack', 'run']
    spec = dict(steps=1000, stride=5, tasks=tasks, priority_probability=1/3, sources={'recorder':'exact'})
    jobs = [(task,['original']) for task in tasks] * 2
    calls = []

    def child(command, *unused):
        calls.append(command)
        folder = Path(command[command.index('--output')+1])
        folder.mkdir(parents=True)
        (folder/'data.pt').write_bytes(b'exact data')
        if any(command_item.endswith('dagger_long_collect.py') for command_item in command):
            (folder/'summary.json').write_bytes(b'{}')
            (folder/'long_replay_collection.json').write_bytes(b'{}')

    monkeypatch.setattr(pipeline, 'child', child)
    replay = pipeline.collect_round(a,tmp_path,tmp_path,lambda *_: {'strength':1.},
                                    jobs,5,1.,str(actor),{},long_replay=spec)
    assert len(replay) == 6
    assert [c[c.index('--steps')+1] for c in calls] == ['250']*3 + ['1000']*3
    assert [c[c.index('--seed')+1] for c in calls] == [str(i) for i in range(50042,50048)]
    assert all(c[c.index('--design-ids')+1]=='original' for c in calls)
    assert all(c[c.index('--student')+1]==str(actor) for c in calls)
    assert [Path(replay[i]).parent.parent.name for i in range(3,6)] == [
        '0003_stairs','0004_backpack','0005_run']
    new_replay = pipeline.collect_round(a,tmp_path,tmp_path,lambda *_: {'strength':1.},
                                       jobs,5,1.,str(actor),{},long_replay=spec)
    assert new_replay == replay and len(calls) == 6

    (Path(replay[3]).parent/'long_replay_collection.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='long replay collection evidence changed'):
        pipeline.collect_round(a,tmp_path,tmp_path,lambda *_: {'strength':1.},
                               jobs,5,1.,str(actor),{},long_replay=spec)
