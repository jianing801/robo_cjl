#!/bin/bash
# BO watchdog — checks if co_design.py is running, restarts if crashed.
# Run via cron: */15 * * * * bash /data/chenjiale/roboparty_train/bo_watchdog.sh >> /tmp/bo_watchdog.log 2>&1

ROOT=/data/chenjiale/roboparty_train
LOG=/tmp/bo_watchdog.log
STUDY_DB="$ROOT/robolab/scripts/tools/co_design_study.db"
BO_LOG="/tmp/co_design_50_v7.log"
TARGET_TRIALS=50

log() { echo "[$(date '+%m-%d %H:%M')] $1"; }

# Check if BO is alive
if pgrep -f "co_design.py" > /dev/null 2>&1; then
    # BO is running — check if training subprocess exists
    if pgrep -f "co_design_train.py" > /dev/null 2>&1; then
        exit 0  # all good
    fi
    # BO running but no training — might be between trials, give it time
    log "BO alive but no training subprocess (maybe between trials)"
    exit 0
fi

# BO is not running — check DB state
log "BO NOT RUNNING! Checking state..."

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate robo_cjl

# Count the remaining usable BO samples, not raw rows. Failed rows are kept in
# Optuna for auditability but must not permanently block the target budget.
REMAINING=$(python3 -c "
import optuna
s = optuna.load_study(study_name='rpo_flat_leg_co_design', storage='sqlite:///$STUDY_DB')
done = [t for t in s.trials if t.state.name == 'COMPLETE']
print(max(0, $TARGET_TRIALS - len(done)))
" 2>/dev/null)

if [ -z "$REMAINING" ] || [ "$REMAINING" = "0" ] || [ "$REMAINING" = "null" ]; then
    log "No remaining trials. BO complete (or DB error)."
    exit 0
fi

# Check last error in BO log
LAST_ERR=$(grep -E "RuntimeError|ModuleNotFoundError|ImportError|NameError|FAILED" "$BO_LOG" 2>/dev/null | tail -3)

EVAL_SCRIPT="$ROOT/robolab/scripts/tools/co_design_eval.py"

# Do not mutate source code from cron. Refuse to launch and leave a diagnostic
# if an old broken evaluator is restored.
if grep -q "_sys.path" "$EVAL_SCRIPT" 2>/dev/null; then
    log "ERROR: evaluator contains _sys.path; source was not modified automatically"
    exit 1
fi

# Fix any RUNNING trials that are actually dead (mark as FAIL)
python3 -c "
import sqlite3
conn = sqlite3.connect('$STUDY_DB')
conn.execute('PRAGMA journal_mode=WAL')
sid = conn.execute(\"SELECT study_id FROM studies WHERE study_name='rpo_flat_leg_co_design'\").fetchone()
if sid:
    sid = sid[0]
    running = conn.execute('SELECT trial_id, number FROM trials WHERE study_id=? AND state=?', (sid, 'RUNNING')).fetchall()
    for tid, tn in running:
        conn.execute('UPDATE trials SET state=? WHERE trial_id=?', ('FAIL', tid))
        print(f'Marked T{tn} as FAIL (orphaned)')
    conn.commit()
conn.close()
" 2>/dev/null

# Restart BO
log "ERROR: $LAST_ERR"
log "RESTARTING BO (remaining: $REMAINING)..."

nohup python "$ROOT/robolab/scripts/tools/co_design.py" \
    --trials "$TARGET_TRIALS" --n-startup-trials 15 \
    --max-iterations 3000 --num-envs 2048 --eval-episodes 1 \
    > "$BO_LOG" 2>&1 &

NEW_PID=$!
log "BO restarted with PID $NEW_PID"
