import argparse
import json
import os
import time
from pathlib import Path
import yaml
import sys

from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine import MgsaStateMachine, Mode
from sensors.gps_reader import GpsReader
from race_mode import RaceModeController
from imu_state import ImuStateReader

class MgsaIO:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.cmd_path = Path(self.paths["command_file"])
        self.state_dir = Path(self.paths["state_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._last_cmd_raw = None
        self._last_cmd_mtime = 0.0

        self._runtime_pidfile = self.state_dir / "runtime.pid"
        self._runtime_pidfile.write_text(str(os.getpid()), encoding="utf-8")

        self.led_enabled = bool(cfg.get("led_strip", {}).get("enabled", False))
        self.hud_enabled = bool(cfg.get("hud", {}).get("enabled", False))

        self._recording = None
        self._record_path = None
        self._record_f = None

        gcfg = cfg.get("gps", {})
        scfg = gcfg.get("serial", {})
        port = scfg.get("port", "/dev/ttyUSB0")
        baud = int(scfg.get("baud", 460800))

        self.gps = GpsReader(port=port, baud=baud)
        self.gps.start()

        # IMU reader (fed by external SEN0142 daemon via imu_state.json)
        self.imu = ImuStateReader(self.state_dir)

        # Race-mode controller (comparison + feedback logic)
        self._race = RaceModeController(cfg, self.paths)

    def on_mode_changed(self, mode: Mode):
        (self.state_dir / "current_mode.txt").write_text(mode.value, encoding="utf-8")

        if mode in (Mode.RECORD_OUTER, Mode.RECORD_INNER):
            self._start_recording(mode.value)
        else:
            self._stop_recording()

        if not self.led_enabled:
            return

        if mode == Mode.IDLE:
            self._led_off()
        elif mode == Mode.RACE:
            self._led_solid("green")
        elif mode == Mode.RECORD_OUTER:
            self._led_solid("blue")
        elif mode == Mode.RECORD_INNER:
            self._led_solid("red")

    def read_command_if_changed(self):
        try:
            st = self.cmd_path.stat()
        except FileNotFoundError:
            return None
        if st.st_mtime <= self._last_cmd_mtime:
            return None
        raw = self.cmd_path.read_text(encoding="utf-8").strip()
        self._last_cmd_mtime = st.st_mtime
        if not raw or raw == self._last_cmd_raw:
            return None
        self._last_cmd_raw = raw
        try:
            return json.loads(raw)
        except Exception:
            return None

    def gps_has_good_fix(self) -> bool:
        max_age = float(self.cfg.get("start_finish", {}).get("max_gps_age_s", 1.0))
        require_rtk = bool(self.cfg.get("start_finish", {}).get("require_rtk_fix", False))
        if require_rtk:
            return self.gps.is_rtk_fix(max_age_s=max_age)
        return self.gps.has_fix(max_age_s=max_age)

    def gps_get_latlon(self):
        try:
            return self.gps.get_latlon()
        except Exception:
            return None

    def gps_get_latlon_wait(self, timeout_s: float = 2.0):
        t0 = time.monotonic()
        last_err = None
        while (time.monotonic() - t0) < timeout_s:
            try:
                return self.gps.get_latlon()
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        raise last_err if last_err else RuntimeError("GPS: no position")

    def gps_get_speed_kmh(self) -> float:
        try:
            return float(self.gps.get_speed_kmh())
        except Exception:
            return 0.0

    def start_finish_confirm(self):
        self._blink_confirm()
        self.on_mode_changed(self._current_mode())

    def _current_mode(self):
        p = self.state_dir / "current_mode.txt"
        if p.exists():
            try:
                return Mode(p.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        return Mode.IDLE

    def start_finish_error(self):
        self._blink_error()

    def _blink_confirm(self):
        fb = self.cfg.get("feedback", {}).get("start_finish_confirm", {})
        blinks = int(fb.get("blinks", 3))
        on_ms = int(fb.get("on_ms", 120))
        off_ms = int(fb.get("off_ms", 120))
        if not self.led_enabled:
            return
        for _ in range(blinks):
            self._led_solid("green")
            time.sleep(on_ms / 1000.0)
            self._led_off()
            time.sleep(off_ms / 1000.0)

    def _blink_error(self):
        fb = self.cfg.get("feedback", {}).get("start_finish_error", {})
        blinks = int(fb.get("blinks", 6))
        on_ms = int(fb.get("on_ms", 80))
        off_ms = int(fb.get("off_ms", 80))
        if not self.led_enabled:
            return
        for _ in range(blinks):
            self._led_solid("red")
            time.sleep(on_ms / 1000.0)
            self._led_off()
            time.sleep(off_ms / 1000.0)

    def _blink_traj_confirm(self):
        fb = self.cfg.get("feedback", {}).get("trajectory_confirm", {})
        blinks = int(fb.get("blinks", 3))
        on_ms = int(fb.get("on_ms", 120))
        off_ms = int(fb.get("off_ms", 120))
        if not self.led_enabled:
            return
        for _ in range(blinks):
            self._led_solid("green")
            time.sleep(on_ms / 1000.0)
            self._led_off()
            time.sleep(off_ms / 1000.0)

    def _blink_traj_error(self):
        fb = self.cfg.get("feedback", {}).get("trajectory_error", {})
        blinks = int(fb.get("blinks", 6))
        on_ms = int(fb.get("on_ms", 80))
        off_ms = int(fb.get("off_ms", 80))
        if not self.led_enabled:
            return
        for _ in range(blinks):
            self._led_solid("red")
            time.sleep(on_ms / 1000.0)
            self._led_off()
            time.sleep(off_ms / 1000.0)

    def _led_solid(self, color: str):
        self._write_runtime_state({"led": {"mode": "solid", "color": color}})

    def _led_off(self):
        self._write_runtime_state({"led": {"mode": "off"}})

    def _read_track_id(self) -> str | None:
        p = self.state_dir / "track_id.txt"
        if not p.exists():
            return None
        s = p.read_text(encoding="utf-8").strip()
        return s if s else None

    def _write_trajectory_status(self, obj: dict):
        p = self.state_dir / "trajectory_status.json"
        tmp = self.state_dir / "trajectory_status.json.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def _write_runtime_state(self, d: dict):
        p = self.state_dir / "runtime_state.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def trajectory_load(self, force: bool = False) -> bool:
        ts = datetime.now().isoformat(timespec="seconds")

        track_id = self._read_track_id()
        if not track_id:
            self._write_trajectory_status({
                "loaded": False,
                "track_id": None,
                "path": None,
                "points_count": 0,
                "columns": [],
                "ts": ts,
                "error": "track_id_missing",
                "message": "No track_id.txt in state_dir",
            })
            self._blink_traj_error()
            return False

        ideal_path = Path(self.paths["tracks_dir"]) / track_id / "ideal.csv"

        if not ideal_path.exists():
            self._write_trajectory_status({
                "loaded": False,
                "track_id": track_id,
                "path": str(ideal_path),
                "points_count": 0,
                "columns": [],
                "ts": ts,
                "error": "ideal_file_missing",
                "message": "ideal.csv not found for this track",
            })
            self._blink_traj_error()
            return False

        try:
            mtime = ideal_path.stat().st_mtime

            # If not force and already have status for same mtime -> skip (optional)
            if not force:
                prev = self._trajectory_status_read_safe()
                if prev and prev.get("loaded") and prev.get("track_id") == track_id and float(prev.get("file_mtime", 0.0)) == float(mtime):
                    self._write_trajectory_status({
                        "loaded": True,
                        "track_id": track_id,
                        "path": str(ideal_path),
                        "points_count": int(prev.get("points_count", 0)),
                        "columns": list(prev.get("columns", [])),
                        "ts": ts,
                        "file_mtime": mtime,
                        "message": "ok (already loaded, unchanged file)",
                    })
                    self._blink_traj_confirm()
                    return True

            # Read CSV: header + count rows
            with ideal_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip()
                if not header:
                    raise RuntimeError("empty_header")

                cols = [c.strip() for c in header.split(",") if c.strip()]
                if len(cols) < 2:
                    raise RuntimeError("bad_header")

                points = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # basic sanity: must have commas
                    if "," not in line:
                        continue
                    points += 1

            if points <= 0:
                raise RuntimeError("no_points")

            self._write_trajectory_status({
                "loaded": True,
                "track_id": track_id,
                "path": str(ideal_path),
                "points_count": points,
                "columns": cols,
                "ts": ts,
                "file_mtime": mtime,
                "message": "ok",
            })
            self._blink_traj_confirm()
            return True

        except Exception as e:
            self._write_trajectory_status({
                "loaded": False,
                "track_id": track_id,
                "path": str(ideal_path),
                "points_count": 0,
                "columns": [],
                "ts": ts,
                "error": "parse_error",
                "message": str(e),
            })
            self._blink_traj_error()
            return False

    def _trajectory_status_read_safe(self) -> dict | None:
        p = self.state_dir / "trajectory_status.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _start_recording(self, mode_name: str):
        self._stop_recording()
        sessions_dir = Path(self.paths["sessions_dir"])
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        folder = sessions_dir / f"{ts}_{mode_name}"
        folder.mkdir(parents=True, exist_ok=True)
        fpath = folder / "events.jsonl"
        self._record_path = folder
        self._record_f = open(fpath, "a", encoding="utf-8")
        self._recording = mode_name
        meta = {
            "type": mode_name,
            "created": ts,
        }
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _stop_recording(self):
        if self._record_f:
            try:
                self._record_f.flush()
                self._record_f.close()
            except Exception:
                pass

        if self._record_path:
            try:
                done = {
                    "done_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "recording": self._recording,
                }
                (self._record_path / "DONE.json").write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        self._record_f = None
        self._recording = None
        self._record_path = None

    def finalize_recording(self):
        self._stop_recording()
        rp = self._record_path
        rec = self._recording

        if rp is None:
            return

        try:
            done = {
                "done_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "recording": rec,
            }
            (rp / "DONE.json").write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def record_tick(self, lap_type: str):
        if not self._record_f:
            return
            pos = self.gps_get_latlon()
        if not pos:
            # no position yet -> don't write point, just exit
            return
        lat, lon = pos

        payload = {
            "t": time.time(),
            "lap_type": lap_type,
            "gps": {"lat": lat, "lon": lon, "fix": self.gps_has_good_fix()},
        }
        self._record_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._record_f.flush()

    def record_event(self, event_type: str, extra: dict | None = None):
        if not self._record_f:
            return
        lat = None
        lon = None
        try:
            lat, lon = self.gps_get_latlon()
        except Exception:
            pass
        payload = {
            "t": time.time(),
            "type": event_type,
            "gps": {"lat": lat, "lon": lon},
            "speed_kmh": self.gps_get_speed_kmh(),
        }
        if extra:
            payload.update(extra)
        self._record_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._record_f.flush()

    def race_tick(self, zone):
        """
        Single race-mode tick.

        Responsibilities:
          - Read sensors (GPS, later IMU)
          - Ask RaceModeController for comparison & feedback
          - Emit compact runtime_state for HUD / LED daemons
        """
        # Ensure trajectory is loaded for current track
        track_id = self._read_track_id()
        if not self._race.ensure_loaded(track_id):
            fb = self._race._feedback_no_trajectory()
        else:
            # GPS
            pos = self.gps_get_latlon()
            if pos:
                lat, lon = pos
            else:
                lat, lon = None, None

            # Current speed from GPS (m/s)
            v_meas_kmh = self.gps_get_speed_kmh()
            v_meas_mps = v_meas_kmh / 3.6

            # IMU: SEN0142 via external daemon -> imu_state.json
            imu_sample = self.imu.get_latest()
            if imu_sample is not None:
                a_lat_meas = imu_sample.a_lat_mps2
                a_lon_meas = imu_sample.a_lon_mps2
                # yaw_rate available as imu_sample.yaw_rate_rad_s (not yet used here)
            else:
                # No fresh IMU data: fall back to neutral (no extra noise added)
                a_lat_meas = 0.0
                a_lon_meas = 0.0

            gps_ok = self.gps_has_good_fix()

            fb = self._race.update(
                lat=lat,
                lon=lon,
                v_meas_mps=v_meas_mps,
                a_lat_meas_mps2=a_lat_meas,
                a_lon_meas_mps2=a_lon_meas,
                gps_fix_ok=gps_ok,
            )

        race_payload = {
            "status": fb.status,
            "reason": fb.reason,
            "s_idx": fb.s_idx,
            "s_m": fb.s_m,
            "d_lat_m": fb.d_lat_m,
            "delta_v_kmh": fb.delta_v_kmh,
            "delta_a_lat_mps2": fb.delta_a_lat_mps2,
        }

        out = {"race": race_payload}

        if self.led_enabled:
            out["led"] = {
                "mode": fb.led_mode,
                "color": fb.led_color,
                "on_ms": fb.led_on_ms,
                "off_ms": fb.led_off_ms,
            }

        if self.hud_enabled:
            hud = {"text": fb.hud_text}
            if fb.hud_arrow:
                hud["arrow"] = fb.hud_arrow
            if fb.hud_warning:
                hud["warning"] = fb.hud_warning
            out["hud"] = hud

        self._write_runtime_state(out)

    def request_idle(self):
        self._stop_recording()
        if self.led_enabled:
            self._led_off()

    def idle_tick(self):
        return

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    Path(cfg["paths"]["data_root"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["sessions_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["track_cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["state_dir"]).mkdir(parents=True, exist_ok=True)

    io = MgsaIO(cfg)
    sm = MgsaStateMachine(cfg, io)

    tick_hz = 20
    dt = 1.0 / tick_hz

    try:
        while True:
            cmd = io.read_command_if_changed()
            if cmd:
                sm.handle_command(cmd)
            sm.tick()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            io.gps.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
