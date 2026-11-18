"""Verify exact partition guarantees independently of search objective."""
from collections import Counter
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import construct,select_grid,optimize,features,objective
from audit import audit


def bank():
    grid=json.loads((HERE.parent/'configs/rough.json').read_text())['grid']
    selected=select_grid(grid,grid['knee_pool'],grid['ankle_pool'])
    return dict(designs=selected['designs'],ranges_m={'thigh':grid['thigh_range_m'],'calf':grid['calf_range_m']})


@pytest.mark.parametrize('seed',[0,42,43,44,999])
def test_all_pairs_and_both_leg_axes_are_exactly_balanced(seed):
    b=bank();groups=construct(b,seed);rows={d['id']:d for d in b['designs']}
    assert len(groups)==50
    assert Counter(i for g in groups for i in g)==Counter(rows.keys())
    for g in groups:
        assert len(g)==50
        assert set(Counter((rows[i]['knee_motor'],rows[i]['ankle_motor']) for i in g).values())=={2}
        for axis in ('thigh_m','calf_m'):
            counts=Counter(rows[i][axis] for i in g)
            assert len(counts)==10 and set(counts.values())=={5}


def test_reproducibility_and_independent_rejection():
    b=bank();g=construct(b,42)
    assert g==construct(b,42) and g!=construct(b,43)
    g[0][0]=g[1][0]
    with pytest.raises(ValueError,match='Duplicate'):
        audit(g,b,windows=())


def test_search_preserves_groups_and_never_returns_worse_best():
    b=bank();g=construct(b,42)
    final,trace=optimize(g,b,seed=42,iterations=100)
    assert Counter(tuple(v) for v in final)==Counter(tuple(v) for v in g)
    assert objective(features(final,b))<=objective(features(g,b))+1e-12
    assert trace['best']<=trace['initial']
