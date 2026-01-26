import argparse
import json
import time
from pathlib import Path
import yaml

from led_strip import LedStrip

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    state_dir = Path(cfg["paths"]["state_dir"])
    state_file = state_dir / "runtime_state.json"

    led = LedStrip(cfg)
    last_raw = None
    last_mtime = 0.0
    last_led_key = None

    try:
        while True:
            try:
                st = state_file.stat()
                if st.st_mtime > last_mtime:
                    raw = state_file.read_text(encoding="utf-8").strip()
                    last_mtime = st.st_mtime
                    if raw and raw != last_raw:
                        last_raw = raw
                        d = json.loads(raw)
                        led_cfg = d.get("led")
                        if isinstance(led_cfg, dict):
                            key = json.dumps(led_cfg, sort_keys=True)
                            if key != last_led_key:
                                last_led_key = key
                                mode = str(led_cfg.get("mode", "off"))
                                color = str(led_cfg.get("color", "green"))
                                on_ms = int(led_cfg.get("on_ms", 120))
                                off_ms = int(led_cfg.get("off_ms", 120))
                                led.set(mode=mode, color=color, on_ms=on_ms, off_ms=off_ms)
            except FileNotFoundError:
                pass
            except Exception:
                pass

            led.tick()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        led.close()

if __name__ == "__main__":
    main()
