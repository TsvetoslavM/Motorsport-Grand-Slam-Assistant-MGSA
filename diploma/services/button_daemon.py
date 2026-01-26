import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml
from gpiozero import Button

CFG_PATH = "/home/cveto-msga/mgsa/diploma/config/mgsa.yaml"

def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    state_dir = Path(cfg["paths"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg["paths"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)

def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def read_pid(pid_file: Path):
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def write_pid(pid_file: Path, pid: int):
    pid_file.write_text(str(pid), encoding="utf-8")

def start_runtime_if_needed(cfg):
    state_dir = Path(cfg["paths"]["state_dir"])
    pid_file = state_dir / "runtime.pid"
    pid = read_pid(pid_file)
    if pid and pid_is_running(pid):
        return

    runtime_path = Path(cfg["paths"]["project_root"]) / "runtime" / "app.py"
    if not runtime_path.exists():
        return

    p = subprocess.Popen(
        ["python3", "-u", str(runtime_path), "--config", CFG_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    write_pid(pid_file, p.pid)

def write_command(cfg, cmd: dict):
    cmd_file = Path(cfg["paths"]["command_file"])
    tmp = cmd_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(cmd, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cmd_file)

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

class PressTracker:
    def __init__(self):
        self.t0 = None

def main():
    cfg = load_cfg()
    ensure_dirs(cfg)

    bcfg = cfg["buttons"]
    pull_up = (bcfg.get("pull", "up") == "up")
    debounce_ms = float(bcfg.get("debounce_ms", 50))
    long_press_s = float(bcfg.get("long_press_s", 5.0))

    pins = bcfg["gpio"]
    pin_outer = int(pins["outer"])
    pin_inner = int(pins["inner"])
    pin_race = int(pins["race_start"])

    btn_outer = Button(pin_outer, pull_up=pull_up, bounce_time=debounce_ms / 1000.0)
    btn_inner = Button(pin_inner, pull_up=pull_up, bounce_time=debounce_ms / 1000.0)
    btn_race = Button(pin_race, pull_up=pull_up, bounce_time=debounce_ms / 1000.0)

    race_tracker = PressTracker()

    # === TOGGLE STATE (само за тестове) ===
    # Пази последния "mode" който сме пратили към runtime.
    # Ако натиснеш същия бутон втори път -> пращаме idle.
    last_mode = {"mode": "idle"}  # mutable container за closures

    def set_mode_toggle(target_mode: str):
        start_runtime_if_needed(cfg)

        if last_mode["mode"] == target_mode:
            new_mode = "idle"
        else:
            new_mode = target_mode

        last_mode["mode"] = new_mode
        write_command(cfg, {"ts": now_iso(), "type": "mode", "mode": new_mode})

    def on_outer():
        set_mode_toggle("record_outer")

    def on_inner():
        set_mode_toggle("record_inner")

    def on_race_pressed():
        race_tracker.t0 = time.monotonic()

    def on_race_released():
        t0 = race_tracker.t0
        race_tracker.t0 = None
        if t0 is None:
            return
        dt = time.monotonic() - t0

        start_runtime_if_needed(cfg)

        if dt >= long_press_s:
            write_command(cfg, {"ts": now_iso(), "type": "start_finish", "action": "set"})
        else:
            # race няма toggle тук (ако искаш toggle и за race -> казваш)
            last_mode["mode"] = "race"
            write_command(cfg, {"ts": now_iso(), "type": "mode", "mode": "race"})

    btn_outer.when_pressed = on_outer
    btn_inner.when_pressed = on_inner
    btn_race.when_pressed = on_race_pressed
    btn_race.when_released = on_race_released

    stop = False

    def handle_sig(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop:
        time.sleep(0.1)
    btn_outer.when_pressed = on_outer
    btn_inner.when_pressed = on_inner
    btn_race.when_pressed = on_race_pressed
    btn_race.when_released = on_race_released

    stop = False
    def handle_sig(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
