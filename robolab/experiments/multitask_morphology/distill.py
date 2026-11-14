"""Train one shared actor using three frozen experts: BC followed by DAgger."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    engine = Path(__file__).resolve().parent/'shared_policy'
    sys.path.insert(0, str(engine))
    runpy.run_path(str(engine/'pipeline.py'), run_name='__main__')
