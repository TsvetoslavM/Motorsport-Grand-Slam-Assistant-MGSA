import time
from dataclasses import dataclass

import board
import neopixel

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def _parse_color(name: str):
    n = (name or "").strip().lower()
    if n == "red":
        return (255, 0, 0)
    if n == "green":
        return (0, 255, 0)
    if n == "blue":
        return (0, 0, 255)
    if n == "white":
        return (255, 255, 255)
    if n == "yellow":
        return (255, 255, 0)
    if n == "cyan":
        return (0, 255, 255)
    if n == "magenta":
        return (255, 0, 255)
    return (0, 0, 0)

def _board_pin_from_gpio(gpio: int):
    m = {
        10: board.D10, 11: board.D11, 12: board.D12, 13: board.D13, 14: board.D14, 15: board.D15,
        16: board.D16, 17: board.D17, 18: board.D18, 19: board.D19,
        20: board.D20, 21: board.D21, 22: board.D22, 23: board.D23, 24: board.D24, 25: board.D25,
        26: board.D26, 27: board.D27
    }
    if gpio not in m:
        raise ValueError(f"Unsupported GPIO for Blinka mapping: GPIO{gpio}")
    return m[gpio]

@dataclass
class LedState:
    mode: str = "off"
    color: str = "green"
    on_ms: int = 120
    off_ms: int = 120

class LedStrip:
    def __init__(self, cfg: dict):
        lc = cfg.get("led_strip", {})
        self.enabled = bool(lc.get("enabled", False))

        self.count = int(lc.get("count", 30))
        self.gpio_pin = int(lc.get("gpio_pin", 12))
        self.brightness = _clamp01(float(lc.get("brightness", 0.35)))
        self.pixel_order = str(lc.get("pixel_order", "GRB")).strip().upper()

        self.pixels = None
        self.state = LedState()
        self._blink_on = False
        self._t_next = 0.0

        if not self.enabled:
            return

        pin = _board_pin_from_gpio(self.gpio_pin)

        order_map = {
            "RGB": neopixel.RGB,
            "GRB": neopixel.GRB,
            "RGBW": neopixel.RGBW,
            "GRBW": neopixel.GRBW,
        }
        order = order_map.get(self.pixel_order, neopixel.GRB)

        self.pixels = neopixel.NeoPixel(
            pin,
            self.count,
            brightness=self.brightness,
            auto_write=False,
            pixel_order=order,
        )
        self.off()

    def _fill(self, rgb):
        if not self.enabled or self.pixels is None:
            return
        self.pixels.fill(rgb)
        self.pixels.show()

    def set(self, mode: str, color: str = "green", on_ms: int = 120, off_ms: int = 120):
        self.state = LedState(mode=str(mode or "off"), color=str(color or "green"), on_ms=int(on_ms), off_ms=int(off_ms))
        self._blink_on = False
        self._t_next = 0.0
        if self.state.mode == "solid":
            self._fill(_parse_color(self.state.color))
        elif self.state.mode == "off":
            self.off()

    def off(self):
        self.state = LedState(mode="off", color="black")
        self._fill((0, 0, 0))

    def tick(self):
        if not self.enabled or self.pixels is None:
            return
        if self.state.mode != "blink":
            return
        now = time.monotonic()
        if self._t_next == 0.0:
            self._blink_on = True
            self._fill(_parse_color(self.state.color))
            self._t_next = now + (self.state.on_ms / 1000.0)
            return
        if now < self._t_next:
            return
        if self._blink_on:
            self._blink_on = False
            self._fill((0, 0, 0))
            self._t_next = now + (self.state.off_ms / 1000.0)
        else:
            self._blink_on = True
            self._fill(_parse_color(self.state.color))
            self._t_next = now + (self.state.on_ms / 1000.0)

    def close(self):
        try:
            self.off()
        except Exception:
            pass
