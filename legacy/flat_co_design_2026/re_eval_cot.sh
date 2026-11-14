#!/bin/bash
# Final re-eval: per-speed CoT + physically-correct total CoT
# CoT_total = sum(E_i) / (m·g·sum(d_i)); this also handles early termination.
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate robo_cjl

ROOT=/data/chenjiale/roboparty_train
DB="$ROOT/robolab/scripts/tools/co_design_study.db"
EVAL="$ROOT/robolab/scripts/tools/co_design_eval.py"
OUT=/tmp/final_cot_reeval.txt
GPU=2

echo "=== Per-speed CoT re-eval started $(date) ===" > $OUT

# 3000代 40 trials + 12000代 top10（真11999 checkpoint）
TRIALS=$(python3 -c "
import optuna, os
s = optuna.load_study(study_name='rpo_flat_leg_co_design', storage='sqlite:///$DB')
rows = []
for t in s.trials:
    if t.state.name == 'COMPLETE':
        ckpt = t.user_attrs.get('ckpt_path','') if t.user_attrs else ''
        if ckpt and os.path.exists(ckpt):
            rows.append(f\"{t.number}|{t.params['thigh_length']}|{t.params['calf_length']}|{ckpt}|3000\")
# 12000代 top10
top10 = {
  'T10': ('0.2835','0.2897','2026-08-07_01-41-25'),
  'T26': ('0.3000','0.3147','2026-07-28_22-59-46'),
  'T22': ('0.3000','0.3600','2026-08-07_09-36-12'),
  'T30': ('0.2500','0.3000','2026-07-28_11-04-24'),
  'T0':  ('0.2732','0.3385','2026-07-29_15-07-41'),
  'T47': ('0.2576','0.3468','2026-07-29_23-19-20'),
  'T14': ('0.2626','0.3577','2026-08-06_17-39-32'),
  'T24': ('0.3000','0.3342','2026-07-27_10-54-41'),
  'T46': ('0.2440','0.3599','2026-07-29_07-15-45'),
  'T35': ('0.2722','0.3515','2026-07-30_07-07-51'),
}
for k,(th,ca,d) in top10.items():
    tn = k[1:]
    p = f'/data/chenjiale/roboparty_train/logs/rsl_rl/rpo_flat/{d}/model_11999.pt'
    if os.path.exists(p):
        rows.append(f'{tn}|{th}|{ca}|{p}|12000')
print('\n'.join(rows))
" 2>/dev/null)

TOTAL=$(echo "$TRIALS" | grep -c '|')
echo "Total: $TOTAL trials to evaluate" >> $OUT

echo "$TRIALS" | while IFS='|' read -r tn thigh calf ckpt iters; do
    [ -z "$tn" ] && continue
    echo "--- T$tn (iters=$iters) ---" >> $OUT
    R=""
    sumE=0; sumD=0; MASS=""; n=0
    for cmd in "0.5,0,0" "1.0,0,0" "-0.5,0,0"; do
        RESULT=$(CUDA_VISIBLE_DEVICES=$GPU timeout 300 python $EVAL \
            --checkpoint "$ckpt" --thigh "$thigh" --calf "$calf" \
            --task RPO-Flat --num-episodes 1 --num-envs 1 \
            --command="$cmd" --headless 2>&1 | grep "RESULT:")
        if [ -n "$RESULT" ]; then
            RTN=$(echo "$RESULT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read().replace('RESULT: ',''));print(d['avg_episode_return'])")
            COT=$(echo "$RESULT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read().replace('RESULT: ',''));print(d.get('cot','nan'))")
            ENERGY=$(echo "$RESULT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read().replace('RESULT: ',''));print(d.get('total_energy_j','nan'))")
            DIST=$(echo "$RESULT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read().replace('RESULT: ',''));print(d.get('total_distance_m','nan'))")
            MASS=$(echo "$RESULT" | python3 -c "import sys,json;d=json.loads(sys.stdin.read().replace('RESULT: ',''));print(d.get('mass_kg','nan'))")
            echo "  cmd=$cmd rtn=$RTN cot=$COT E=$ENERGY D=$DIST" >> $OUT
            R="$R|$RTN|$COT"
            sumE=$(python3 -c "print($sumE + $ENERGY)")
            sumD=$(python3 -c "print($sumD + $DIST)")
            n=$((n+1))
        else
            echo "  cmd=$cmd FAILED" >> $OUT
            R="$R|nan|nan"
        fi
    done
    # CoT_total = sum(energy) / (m*g*sum(distance)). This remains correct
    # if an episode terminates early, unlike summing mean powers/speeds.
    if [ $n -eq 3 ]; then
        TOTCOT=$(python3 -c "m=float('$MASS'); e=float('$sumE'); d=float('$sumD'); print(f'{e/(m*9.81*d):.4f}' if d>0 else 'nan')")
    else
        TOTCOT="nan"
    fi
    echo "RESULT_ROW: T$tn|$iters|$thigh|$calf|$sumE|$sumD|$TOTCOT${R}" >> $OUT
done

echo "=== DONE $(date) ===" >> $OUT
