"""Completa solo CarRacing mancante e aggiorna il report, senza chiamate a Codex."""
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from uuid import uuid4

from run_experiments import ROOT, jobs, save

RUNS = ROOT / "outputs" / "experiments_20261014"
NAMES = ("racing_dqn", "racing_ppo")
STATUS = RUNS / "status_remaining.json"
STATE = RUNS / "background_state.json"


def now():
    return datetime.now().isoformat(timespec="seconds")


def all_statuses():
    states = {}
    for path in sorted(RUNS.glob("status_*.json")):
        for name, value in json.loads(path.read_text(encoding="utf-8")).items():
            old = states.get(name, {})
            if (not old or value["status"] == "completed"
                    or old["status"] != "completed" and value.get("started", "") >= old.get("started", "")):
                states[name] = value
    return states


def completed_on_disk(name, record):
    path = Path(record["output"]) if record.get("output") else None
    if path is None:
        return False
    if name.endswith("_eval"):
        return (path / "evaluation.json").is_file() and (path / "video/episode_0.mp4").is_file()
    metrics = path / "metrics.csv"
    if not (path / "last.pt").is_file() or not metrics.is_file():
        return False
    with metrics.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    key, target = ("episode", 500) if name == "racing_dqn" else ("steps", 1000000)
    return bool(rows) and int(float(rows[-1][key])) >= target


def update_report():
    result = subprocess.run([sys.executable, str(ROOT / "report_experiments.py")], cwd=ROOT,
                            stdout=sys.stdout, stderr=sys.stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        print(f"{now()} Aggiornamento report non riuscito: {result.returncode}", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    # Lock del sistema operativo, rilasciato anche se il processo viene terminato.
    import msvcrt
    lock = (RUNS / ".remaining.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    sys.stdout = (RUNS / "background.log").open("a", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    print(f"{now()} Avvio coordinatore indipendente PID {os.getpid()}", flush=True)
    states = all_statuses()
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    selected = {name: states.get(name, dict(status="pending"))
                for base in NAMES for name in (base, base + "_eval")}
    for name, record in selected.items():
        if record["status"] != "completed" and completed_on_disk(name, record):
            record = {**record, "status": "completed", "recovered_from_artifacts": True}
        previous[name] = record
    save(STATUS, previous)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HOME=str(ROOT / "data/huggingface"), TQDM_DISABLE="1",
                       HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    active = {}
    definitions = {name: (module, raw) for name, module, raw in jobs("gpu")}

    def launch(name, source=None):
        if previous[name]["status"] == "completed":
            return
        output = RUNS / name / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
        output.mkdir(parents=True)
        if name.endswith("_eval"):
            base = name.removesuffix("_eval")
            tokens = ["--checkpoint", str(Path(previous[base]["output"]) / "best.pt"),
                      "--video", "--eval-episodes", "20"]
        else:
            _, raw = definitions[name]
            tokens = [*raw.split(), "--continue-run", str(source)]
        command = [sys.executable, "-m", "lab3.ex3_3", *tokens, "--device", "cuda", "--output-dir", str(output)]
        record = dict(status="running", command=command, started=now(), output=str(output))
        if source is not None:
            record["continuation_source"] = str(source)
        log = (output / "run.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            log.close()
            raise
        record["pid"] = process.pid
        previous[name] = record
        active[name] = (process, log, time.monotonic())
        save(STATUS, previous)
        print(f"{now()} START {name}, PID {process.pid}, {output}", flush=True)

    awake = os.name == "nt"
    try:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        save(STATE, dict(status="running", pid=os.getpid(), started=now(), jobs=list(selected)))
        for base in NAMES:
            record = previous[base]
            if record["status"] == "completed":
                launch(base + "_eval")
            else:
                source = Path(record["output"])
                if not (source / "best.pt").is_file():
                    raise FileNotFoundError(f"Checkpoint da continuare non trovato: {source}")
                launch(base, source)
        update_report()
        refreshed = time.monotonic()
        while active:
            for name, (process, log, start) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                record = previous[name]
                record.update(status="completed" if code == 0 else "failed", returncode=code,
                              finished=now(), duration_seconds=round(time.monotonic() - start, 2))
                del active[name]
                if name in NAMES:
                    if code == 0:
                        launch(name + "_eval")
                    else:
                        previous[name + "_eval"] = dict(status="blocked", dependency=name)
                save(STATUS, previous)
                print(f"{now()} END {name}: {record['status']}", flush=True)
                update_report()
            if time.monotonic() - refreshed >= 120:
                update_report()
                refreshed = time.monotonic()
            time.sleep(1)
        success = all(record["status"] == "completed" for record in previous.values())
        save(STATE, dict(status="completed" if success else "failed", pid=os.getpid(), finished=now(),
                         jobs={name: record["status"] for name, record in previous.items()}))
        print(f"{now()} Terminato. Report: {ROOT / 'RISULTATI.md'}", flush=True)
    except BaseException as error:
        save(STATE, dict(status="failed", pid=os.getpid(), finished=now(), error=repr(error)))
        print(f"{now()} ERRORE: {error!r}", flush=True)
        raise
    finally:
        update_report()
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == "__main__":
    main()
