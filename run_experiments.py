"""Training ed evaluation reali, con log, dipendenze e ripartenza dopo errori.

Eseguire con il Python dell'ambiente DLA. GPU e CPU hanno code indipendenti.
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def jobs(group):
    if group == "ood_extra":
        return [("clean_control_10", "lab4.ex2_2", "--checkpoint @ood_intro/training/best.pt --mode clean --epochs 10"),
                ("clean_control_5", "lab4.ex2_2", "--checkpoint @ood_intro/training/best.pt --mode clean --epochs 5")]
    if group in ("racing_dqn", "racing_ppo"):
        return [job for job in jobs("gpu") if job[0] in (group, group + "_eval")]
    if group == "nlp_extra":
        return [("emoji_finetune", "lab2.ex3_1", "--task emoji --fine-tune --epochs 5 --batch-size 32")]
    if group == "vision":
        names = {"cnn", "cnn_eval", "gradients", "gradients_eval", "gradcam"}
        return [job for job in jobs("gpu") if job[0] in names]
    if group == "gpu":
        return [
            ("mlp", "lab1.ex1_1", "--epochs 20 --batch-size 256"),
            ("mlp_eval", "lab1.evaluate", "--checkpoint @mlp --batch-size 256"),
            ("cnn", "lab1.ex1_2", "--epochs 20 --batch-size 128"),
            ("cnn_eval", "lab1.evaluate", "--checkpoint @cnn --batch-size 256"),
            ("gradients", "lab1.ex2_1", "--epochs 5 --batch-size 128"),
            ("gradients_eval", "lab1.evaluate", "--checkpoint @gradients --batch-size 256"),
            ("fcn", "lab1.ex2_2", "--epochs 10 --batch-size 128"),
            ("fcn_eval", "lab1.evaluate", "--checkpoint @fcn/classifier/best.pt --batch-size 256"),
            ("gradcam", "lab1.ex2_3", "--checkpoint @cnn/residual_2/best.pt --samples 16"),
            ("dante", "lab2.ex1", "--epochs 10 --steps-per-epoch 250 --batch-size 64"),
            ("gpt2_tokens", "lab2.ex2_1", ""),
            ("gpt2_generation", "lab2.ex2_2", ""),
            ("irony", "lab2.ex3_1", "--task irony --epochs 5 --batch-size 64"),
            ("irony_finetune", "lab2.ex3_1", "--task irony --fine-tune --epochs 5 --batch-size 32"),
            ("emoji", "lab2.ex3_1", "--task emoji --epochs 5 --batch-size 64"),
            ("arc", "lab2.ex3_2", "--dataset arc --epochs 4 --batch-size 16"),
            ("swag", "lab2.ex3_2", "--dataset swag --epochs 2 --batch-size 16"),
            ("scifact", "lab2.ex3_3", "--epochs 5 --batch-size 16"),
            ("ood_intro", "lab4.intro", "--epochs 10 --snapshot-epochs 1 5 10 --batch-size 128"),
            ("ood_pipeline", "lab4.ex1_1", "--checkpoint @ood_intro/training/best.pt"),
            ("ood_metrics", "lab4.ex1_2", "--checkpoint @ood_intro/training/best.pt --ood fake"),
            ("fgsm", "lab4.ex2_1", "--checkpoint @ood_intro/training/best.pt"),
            ("adversarial", "lab4.ex2_2", "--checkpoint @ood_intro/training/best.pt --epochs 10"),
            ("odin", "lab4.ex3_1", "--checkpoint @ood_intro/training/best.pt"),
            ("jarn", "lab4.ex3_2", "--checkpoint @ood_intro/training/best.pt --epochs 5"),
            ("targeted", "lab4.ex3_3", "--checkpoint @ood_intro/training/best.pt --target 0"),
            ("racing_dqn", "lab3.ex3_3", "--algorithm dqn --episodes 500 --eval-every 25 --epsilon-steps 100000"),
            ("racing_dqn_eval", "lab3.ex3_3", "--checkpoint @racing_dqn/best.pt --video --eval-episodes 20"),
            ("racing_ppo", "lab3.ex3_3", "--algorithm ppo --total-steps 1000000 --eval-every 50"),
            ("racing_ppo_eval", "lab3.ex3_3", "--checkpoint @racing_ppo/best.pt --video --eval-episodes 20"),
        ]
    return [
        ("cartpole_reinforce", "lab3.ex1", "--episodes 500 --eval-every 50"),
        ("cartpole_baselines", "lab3.ex2", "--episodes 1000 --eval-every 50"),
        ("cartpole_value_eval", "lab3.ex2", "--checkpoint @cartpole_baselines/value/best.pt --video --eval-episodes 100"),
        ("lunar_reinforce", "lab3.ex3_1", "--episodes 2000 --eval-every 100"),
        ("lunar_value_eval", "lab3.ex3_1", "--checkpoint @lunar_reinforce/value/best.pt --video --eval-episodes 100"),
        ("cartpole_dqn", "lab3.ex3_2", "--env CartPole-v1 --episodes 1000 --eval-every 50"),
        ("cartpole_dqn_eval", "lab3.ex3_2", "--checkpoint @cartpole_dqn/best.pt --video --eval-episodes 100"),
        ("lunar_dqn", "lab3.ex3_2", "--env LunarLander-v3 --episodes 1000 --eval-every 50"),
        ("lunar_dqn_eval", "lab3.ex3_2", "--checkpoint @lunar_dqn/best.pt --video --eval-episodes 100"),
    ]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    # Su Windows antivirus e watcher possono tenere brevemente aperto il JSON.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["cpu", "gpu", "vision", "nlp_extra", "racing_dqn", "racing_ppo", "ood_extra"], required=True)
    p.add_argument("--root", type=Path, default=ROOT / "outputs" / "experiments_20261014")
    p.add_argument("--jobs", nargs="+", default=["*"], help="Nomi o pattern; i job riusciti vengono saltati")
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / f"status_{args.group}.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    # Una ripetizione riuscita nella coda vision vale anche per i job GPU
    # omonimi, senza ripetere training già completati nella stessa radice.
    for other_path in root.glob("status_*.json"):
        if other_path == status_path:
            continue
        other = json.loads(other_path.read_text(encoding="utf-8"))
        for name, value in other.items():
            if value.get("status") == "completed" and status.get(name, {}).get("status") != "completed":
                status[name] = value
    selected = [job for job in jobs(args.group) if any(fnmatch.fnmatchcase(job[0], pattern) for pattern in args.jobs)]
    for name, module, raw in selected:
        status.setdefault(name, dict(status="pending", module=module, arguments=raw))
    save(status_path, status)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HOME=str(ROOT / "data" / "huggingface"), HF_HUB_DISABLE_PROGRESS_BARS="1",
                       TQDM_DISABLE="1", TOKENIZERS_PARALLELISM="false")
    for name, module, raw in selected:
        if status.get(name, {}).get("status") == "completed":
            print(f"SKIP {name}: già completato", flush=True)
            continue
        output = root / name / datetime.now().strftime("%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=False)
        tokens = []
        blocked = None
        for token in raw.split():
            if token.startswith("@"):
                dependency, _, suffix = token[1:].partition("/")
                if status.get(dependency, {}).get("status") != "completed":
                    blocked = dependency
                    break
                token = str(Path(status[dependency]["output"]) / suffix)
            tokens.append(token)
        if blocked:
            status[name] = dict(status="blocked", dependency=blocked, output=str(output))
            save(status_path, status)
            print(f"BLOCKED {name}: {blocked}", flush=True)
            continue
        command = [sys.executable, "-m", module, *tokens, "--device", "cpu" if args.group == "cpu" else "cuda",
                   "--output-dir", str(output)]
        start = time.time()
        status[name] = dict(status="running", command=command, started=datetime.now().isoformat(), output=str(output))
        save(status_path, status)
        print(f"START {name}: {' '.join(command)}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status[name].update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode,
                            duration_seconds=round(time.time() - start, 2), finished=datetime.now().isoformat())
        save(status_path, status)
        print(f"END {name}: {status[name]['status']}, {status[name]['duration_seconds']}s", flush=True)
        if result.returncode:
            print((output / "run.log").read_text(encoding="utf-8")[-4000:], flush=True)
    print(f"Registro: {status_path}", flush=True)
    if any(status[name]["status"] != "completed" for name, _, _ in selected):
        sys.exit(1)


if __name__ == "__main__":
    main()
