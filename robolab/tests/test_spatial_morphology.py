import pytest
import torch
from robolab.spatial_morphology import make_layout, balanced_indices, envelope_limit, interpolate_validated_envelope


def bank():
    return {"designs": [dict(id=str(i), split="train", context=[0.]*22,
        graph={"joint_names": ["knee", "ankle"]}) for i in range(20)] +
        [dict(id="holdout", split="holdout")]}


def test_layout_all_20_no_holdout_2048():
    layout = make_layout(bank())
    assert len(layout["env_design_index"]) == 2048
    assert sorted(layout["counts"].values()) == [102]*12 + [103]*8
    assert "holdout" not in layout["design_ids"]
    assert not layout["replicate_physics"] and not layout["random_asset_choice"]


def test_reject_holdout_duplicates_and_incompatible():
    with pytest.raises(KeyError):
        make_layout(bank(), design_ids=["holdout"])
    with pytest.raises(ValueError):
        make_layout(bank(), design_ids=["1", "1"])
    b = bank()
    b["designs"][0]["graph"]["joint_names"] = ["other"]
    with pytest.raises(ValueError):
        make_layout(b)


def test_balanced_onpolicy_exact_coverage_each_epoch():
    groups = torch.arange(2048) % 20
    batches = balanced_indices(groups, 24, 4)
    flat = torch.cat(batches)
    assert torch.equal(flat.sort().values, torch.arange(2048*24))
    assert {len(x) for x in batches} == {2048*24//4}
    for group in range(20):
        counts = [int((groups[x % 2048] == group).sum()) for x in batches]
        assert max(counts)-min(counts) <= 1


def test_motor_curve_distinct_and_symmetric():
    speed = torch.tensor([[-30., -5., 0., 5., 30.]])
    a = envelope_limit(speed, [[0.,10.],[10.,0.]], 8.)
    b = envelope_limit(speed, [[0.,20.],[20.,0.]], 20.)
    assert torch.allclose(a, torch.tensor([[0.,5.,8.,5.,0.]]))
    assert torch.allclose(b, torch.tensor([[0.,15.,20.,15.,0.]]))
    with pytest.raises(ValueError):
        envelope_limit(speed, [[0.,1.],[0.,2.]], 1.)


def test_cached_hot_path_identical_to_validating_envelope():
    speed = torch.randn(2048, 4) * 30
    table = torch.tensor([[0., 20.], [3., 16.], [8., 10.], [20., 0.]])
    assert torch.equal(envelope_limit(speed, table, 18.),
                       interpolate_validated_envelope(speed, table, 18.))
