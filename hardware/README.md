## Hardware / embedded runtime

The `hardware/` folder contains the **on‑car MGSA runtime** (what you place on the device), based on the code under `diploma/`.
It includes:

- `diploma/runtime/` – main state machine and IO:
  - `app.py` – core loop, integrates GPS, IMU, lap recording and race‑mode controller.
  - `state_machine.py` – high‑level modes (idle, record outer/inner, race).
  - `race_mode.py` – comparison to ideal trajectory and feedback generation.
  - `imu_state.py` – IMU state reader.
- `diploma/services/button_daemon.py` – GPIO button daemon (outer/inner/race and start‑finish control).
- `diploma/sensors/gps_reader.py` – GPS reader for the on‑board receiver.
- `diploma/hud/led_strip_daemon.py`, `led_strip.py` – LED HUD feedback on the car.
- `diploma/net/` – upload/download helpers for laps and tracks.
- `diploma/config/mgsa.yaml` – main configuration file (paths, GPIO pins, feedback timings, etc.).

Session and state data (CSV laps, JSON state, etc.) are stored under `diploma/data/`.

### Typical deployment flow (high‑level)

1. Copy the repo (or just `diploma/` + config) to the device.
2. Install Python 3 and required packages (GPIO, PyYAML, etc.).
3. Configure `diploma/config/mgsa.yaml` for:
   - `paths.*` – data/state root on the device.
   - `buttons.gpio.*` – GPIO pins for outer/inner/race buttons.
   - `gps.serial.*` – GPS serial port and baudrate.
4. Start the runtime and services (examples, run from `hardware/` root on device):

```bash
python -m diploma.runtime.app --config diploma/config/mgsa.yaml
python -m diploma.services.button_daemon &
python -m diploma.hud.led_strip_daemon &
```

Exact service orchestration (systemd, supervisor, etc.) is up to the target platform.

