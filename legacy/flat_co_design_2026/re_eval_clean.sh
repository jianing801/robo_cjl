#!/bin/bash
# Final clean eval: nominal params + nominal initial state + heading tracking
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate robo_cjl

ROOT=/data/chenjiale/roboparty_train
DB="$ROOT/robolab/scripts/tools/co_design_study.db"
EVAL="$ROOT/robolab/scripts/tools/co_design_eval.py"
OUT=/tmp/clean_reeval_results.txt
GPU=3

echo "=== Clean eval (nominal + heading) started $(date) ===" > $OUT

TRIALS=$(python3 -c "
import optuna, os
s = optuna.load_study(study_name='rpo_flat_leg_co_design', storage='sqlite:///$DB')
for t in s.trials:
    if t.state.name == 'COMPLETE':
        ckpt = t.user_attrs.get('ckpt_path','') if t.user_attrs else ''
        if ckpt and os.path.exists(ckpt):
            print(f'{t.number}|{t.params[\"thigh_length\"]}|{t.params[\"calf_length\"]}|{ckpt}')
" 2>/dev/null)

while IFS='|' read -r tn thigh calf ckpt; do
    [ -z "$tn" ] && continue
    echo "--- T$tn ---" >> $OUT
    sum_r=0; sum_e=0; sum_d=0; mass=""; n=0
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
            sum_r=$(python3 -c "print($sum_r + $RTN)")
            sum_e=$(python3 -c "print($sum_e + $ENERGY)")
            sum_d=$(python3 -c "print($sum_d + $DIST)")
            n=$((n+1))
        else
            echo "  cmd=$cmd FAILED" >> $OUT
        fi
    done
    if [ $n -eq 3 ]; then
        avg_r=$(python3 -c "print(f'{$sum_r/$n:.3f}')")
        total_cot=$(python3 -c "m=float('$mass'); e=float('$sum_e'); d=float('$sum_d'); print(f'{e/(m*9.81*d):.4f}' if d>0 else 'nan')")
        echo "RESULT_ROW: T$tn|$thigh|$calf|$avg_r|$total_cot" >> $OUT
    else
        echo "RESULT_ROW: T$tn|$thigh|$calf|INCOMPLETE ($n/3 commands succeeded)" >> $OUT
    fi
done <<< "$TRIALS"

echo "=== DONE $(date) ===" >> $OUT
