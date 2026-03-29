"""Comma body hardware bridge — translates velocity commands to wheel motors.

Subscribes to ``body/control/velocity`` on Zenoh and forwards the commands
to the comma body's differential-drive wheels.

On the comma device the bridge talks to the body board via cereal / CAN.
On a dev machine it runs in **dry-run** mode (logs commands to stdout).

Usage
-----
    uv run body-driver                          # dry-run (no hardware)
    uv run body-driver --comma 192.168.1.10     # real comma body
"""

from __future__ import annotations

import argparse
import json
import struct
import time

import zenoh

from rerun_prompt_da.motor_controller import VELOCITY_TOPIC

# CAN arbitration IDs for the comma body motor controller.
# These match the body board firmware (see openpilot/selfdrive/body/).
MOTOR_CMD_ADDR = 0x250          # left-speed + right-speed
MOTOR_CMD_BUS = 0

# Speed is sent as int16 in units of ~0.001 m/s (firmware-dependent).
SPEED_SCALE = 1000.0
MAX_SPEED_TICKS = 500           # clamp for safety


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _build_motor_can_msg(left: float, right: float) -> bytes:
    """Pack left/right wheel speeds into an 8-byte CAN payload.

    Format matches the comma body motor controller expectation:
        bytes 0-1: int16 left  (big-endian, units ≈ 0.001 m/s)
        bytes 2-3: int16 right (big-endian, units ≈ 0.001 m/s)
        bytes 4-7: reserved / zero
    """
    lt = int(_clamp(left * SPEED_SCALE, -MAX_SPEED_TICKS, MAX_SPEED_TICKS))
    rt = int(_clamp(right * SPEED_SCALE, -MAX_SPEED_TICKS, MAX_SPEED_TICKS))
    return struct.pack(">hh4x", lt, rt)


class DryRunDriver:
    """Prints velocity commands instead of sending them to hardware."""

    def __init__(self):
        self._last_print = 0.0

    def send(self, left: float, right: float):
        now = time.monotonic()
        if now - self._last_print > 0.25:          # throttle output
            print(f"[dry-run] L={left:+.3f}  R={right:+.3f} m/s")
            self._last_print = now


class CommaBodyDriver:
    """Send wheel commands to the comma body via cereal CAN."""

    def __init__(self, addr: str):
        import cereal.messaging as messaging
        self._pm = messaging.PubMaster(["sendcan"])
        self._addr = addr
        print(f"CommaBodyDriver: connected to {addr}")

    def send(self, left: float, right: float):
        import cereal.messaging as messaging
        from cereal import car

        can_data = _build_motor_can_msg(left, right)

        msg = messaging.new_message("sendcan", 1)
        msg.sendcan[0].address = MOTOR_CMD_ADDR
        msg.sendcan[0].busTime = 0
        msg.sendcan[0].dat = can_data
        msg.sendcan[0].src = MOTOR_CMD_BUS
        self._pm.send("sendcan", msg)


def main():
    parser = argparse.ArgumentParser(description="Comma body motor driver")
    parser.add_argument(
        "--comma", type=str, default=None, metavar="ADDR",
        help="Comma device IP.  Omit for dry-run mode.",
    )
    parser.add_argument(
        "--connect", type=str, default=None,
        help="Zenoh router endpoint (e.g. tcp/localhost:7447)",
    )
    args = parser.parse_args()

    # Select driver backend.
    if args.comma:
        driver = CommaBodyDriver(args.comma)
    else:
        print("No --comma specified, running in dry-run mode")
        driver = DryRunDriver()

    # Zenoh setup.
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    def _on_velocity(sample):
        try:
            msg = json.loads(sample.payload.to_bytes().decode())
            left = float(msg.get("left", 0.0))
            right = float(msg.get("right", 0.0))
            driver.send(left, right)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass

    sub = session.declare_subscriber(VELOCITY_TOPIC, _on_velocity)
    print(f"Body driver: subscribed to '{VELOCITY_TOPIC}'")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping body driver")
    finally:
        sub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
