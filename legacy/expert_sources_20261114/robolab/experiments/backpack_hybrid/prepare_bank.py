"""Backpack-only entry using the verified rotating morphology engine."""
import json,runpy,sys
from pathlib import Path
if __name__=='__main__':
    engine=Path(__file__).resolve().parent.parent/'rough_hybrid'
    sys.path.insert(0,str(engine))
    if '--task-key' in sys.argv:raise ValueError('Backpack entry fixes its task identity')
    sys.argv += ['--task-key','backpack']
    runpy.run_path(str(engine/'prepare_bank.py'),run_name='__main__')
