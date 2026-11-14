"""Re-evaluate all 10 trials with 6 commands using subprocess (one per trial)."""
import json, re, subprocess, sys, optuna

SCRIPT = "/data/chenjiale/roboparty_train/robolab/scripts/tools/co_design_eval.py"
DB = "sqlite:////data/chenjiale/roboparty_train/robolab/scripts/tools/co_design_study.db"
ALL_CMDS = "0.5,0,0;1.0,0,0;-0.5,0,0;0,0,1.0;0,0,-1.0;0,0.5,0"
CMD_KEYS = "0.5,0,0 1.0,0,0 -0.5,0,0 0,0,1.0 0,0,-1.0 0,0.5,0".split()

s = optuna.load_study(study_name="rpo_flat_leg_co_design", storage=DB)
trials = sorted([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.number)

print("T  thigh  calf  " + "  ".join(f"{k:>8s}" for k in CMD_KEYS) + "   avg")
sys.stdout.flush()

for t in trials:
    thigh = t.params["thigh_length"]
    calf  = t.params["calf_length"]
    ckpt  = t.user_attrs.get("ckpt_path", "")
    if not ckpt: continue

    cmd = [sys.executable, SCRIPT, "--checkpoint", ckpt, "--thigh", str(thigh), "--calf", str(calf),
           "--command", ALL_CMDS, "--task", "RPO-Flat", "--num-episodes", "1", "--num-envs", "1", "--headless"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        m = re.search(r"RESULT:\s*(.*)", r.stdout + r.stderr)
        if r.returncode == 0 and m:
            data = json.loads(m.group(1))
            vals = {c["command"]: c["total_return"] for c in data["commands"]}
            row = f"T{t.number} {thigh:.3f} {calf:.3f}"
            for ck in CMD_KEYS:
                row += f"  {vals.get(ck, 0):8.1f}"
            row += f"  {data['avg_return']:6.3f}"
            print(row, flush=True)
        else:
            err = r.stderr[-200:] if r.stderr else ""
            print(f"T{t.number} {thigh:.3f} {calf:.3f}  FAIL rc={r.returncode} {err}", flush=True)
    except Exception as e:
        print(f"T{t.number} {thigh:.3f} {calf:.3f}  ERR:{e}", flush=True)

print("DONE")
