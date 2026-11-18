"""Fine-tune one exact structure and evaluate it independently."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    engine = Path(__file__).resolve().parent/'shared_policy'
    sys.path.insert(0, str(engine))
    runpy.run_path(str(engine/'joint_finetune.py'), run_name='__main__')
