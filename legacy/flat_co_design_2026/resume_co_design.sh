#!/bin/bash
# Resume co-design BO training — handles orphaned trials and continues.
# Usage: bash resume_co_design.sh
set -e

ROOT=/data/chenjiale/roboparty_train
cd $ROOT
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate robo_cjl

DB="robolab/scripts/tools/co_design_study.db"
STUDY="rpo_flat_leg_co_design"
EVAL_SCRIPT="robolab/scripts/tools/co_design_eval.py"

echo "=== Co-Design Resume ==="

# 1. Kill orphaned training procs
for pid in $(pgrep -f "co_design_train.py" 2>/dev/null); do
    ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')
    if [ -z "$ppid" ] || ! ps -p $ppid >/dev/null 2>&1; then
        echo "[resume] Killing orphan train PID=$pid"
        kill -9 $pid 2>/dev/null
    fi
done

# 2. Fix RUNNING trials that are actually complete
if [ -f "$DB" ]; then
python - "$DB" "$STUDY" "$EVAL_SCRIPT" << 'PYEOF'
import optuna, os, re, sys, subprocess, json, time

db_path, study_name, eval_script = sys.argv[1], sys.argv[2], sys.argv[3]
db_url = f"sqlite:///{os.path.abspath(db_path)}"
s = optuna.load_study(study_name=study_name, storage=db_url)

for t in s.trials:
    if t.state != optuna.trial.TrialState.RUNNING:
        continue

    thigh = t.params['thigh_length']
    calf  = t.params['calf_length']
    log_dir = t.user_attrs.get('log_dir', '') if t.user_attrs else ''

    ckpt = None
    if log_dir and os.path.isdir(log_dir):
        ckpts = []
        for name in os.listdir(log_dir):
            match = re.fullmatch(r'model_(\d+)\.pt', name)
            if match:
                ckpts.append((int(match.group(1)), name))
        if ckpts:
            ckpt = os.path.join(log_dir, max(ckpts)[1])

    if not ckpt:
        print(f'[resume] Trial {t.number}: no checkpoint, marking FAIL')
        s.tell(t, float('inf'), state=optuna.trial.TrialState.FAIL)
        continue

    print(f'[resume] Trial {t.number}: ckpt={os.path.basename(ckpt)}, running eval...')
    try:
        # Same protocol as the co_design.py objective: single 1.0 m/s forward
        # by default. Add more commands here (and pass --eval-commands to
        # co_design.py) to enable multi-speed evaluation.
        returns = []
        for cmd in ("1.0,0,0",):
            r = subprocess.run([
                sys.executable, eval_script,
                '--checkpoint', ckpt,
                '--thigh', str(thigh),
                '--calf', str(calf),
                '--task', 'RPO-Flat',
                '--num-episodes', '1',
                '--num-envs', '1',
                f'--command={cmd}',
                '--headless',
            ], capture_output=True, text=True, timeout=7200)
            m = re.search(r'RESULT:\s*(.*)', r.stdout + r.stderr)
            if r.returncode != 0 or not m:
                raise RuntimeError(f"Eval failed for cmd={cmd}")
            data = json.loads(m.group(1))
            returns.append(float(data['avg_episode_return']))
        ep_return = sum(returns) / len(returns)
        t.set_user_attr('ckpt_path', ckpt)
        t.set_user_attr('episode_return', ep_return)
        # Study direction is maximize and the objective is the averaged
        # episode return (co_design_eval.py no longer reports an energy penalty).
        s.tell(t, ep_return, state=optuna.trial.TrialState.COMPLETE)
        per_cmd = ' '.join(f'{x:.3f}' for x in returns)
        print(f'[resume] Trial {t.number}: DONE eval_r={ep_return:.3f} (per-cmd {per_cmd})')
    except Exception as e:
        print(f'[resume] Trial {t.number}: eval error ({e}), marking FAIL')
        s.tell(t, float('inf'), state=optuna.trial.TrialState.FAIL)
PYEOF
fi

# 3. Continue BO
echo "[resume] Starting co-design..."
python robolab/scripts/tools/co_design.py \
    --trials 10 --n-startup-trials 4 \
    --max-iterations 3000 --num-envs 2048 --eval-episodes 1
