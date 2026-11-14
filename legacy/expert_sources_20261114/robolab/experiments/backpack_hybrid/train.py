"""Backpack-only entry using the verified rotating morphology engine."""
import json,runpy,sys
from pathlib import Path
if __name__=='__main__':
    engine=Path(__file__).resolve().parent.parent/'rough_hybrid'
    sys.path.insert(0,str(engine))
    if '--bank' not in sys.argv:raise ValueError('--bank required')
    bank=Path(sys.argv[sys.argv.index('--bank')+1])
    if json.loads(bank.read_text(encoding='utf8')).get('task_key')!='backpack':raise ValueError('Backpack entry requires a backpack bank')
    runpy.run_path(str(engine/'train.py'),run_name='__main__')
