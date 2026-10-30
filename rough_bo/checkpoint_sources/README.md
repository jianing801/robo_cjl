# Recorded source for the 12,000-update Rough checkpoint

These three files are archived bytes from the source tree of `model_11999.pt`
(checkpoint SHA256 `a0cd01d59c47588eb23a52b440beac41d626f7b4a85473fb7135743930face26`).
They are used to build an isolated BO source snapshot. BO executes its own
`bo_finetune_candidate.py` and `bo_finetune_evaluate.py` workers.

| File | Recorded SHA256 |
|---|---|
| persistent_worker.py | 8dc97306cba7812c141fcb72a8233fd288a3ee4be31b31c4d0217a803a71c0c7 |
| train.py | 872cad0dece779c38be133e199904fa541f496a58939b2a693b5890991958558 |
| train_stage.py | 88648149221118542cbb986d14005d0008a5e96fa23d2fc52aab848bc2714108 |

Recovery was checked with the real checkpoint on CPU: the original three
mismatches became 74/74 exact source matches, and the input source files stayed
unchanged. No simulation or policy optimization was run for that check.

The CPU regression suite covers exact restoration, source preservation, cached
snapshot reuse, unknown runtime changes, altered recovery files, altered cached
snapshots and escaping paths, along with foreground launch and log behavior.

```bash
python -m unittest discover -s rough_bo/tests -p 'test_*.py' -v
```
