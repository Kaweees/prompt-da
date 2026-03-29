"""Comma body hardware bridge — translates DGX velocity commands to testJoystick.

Subscribes to ``body/control/velocity`` on Zenoh and converts (linear, angular)
differential-drive velocities into testJoystick (accel, steer) axes that
joystickd understands.

This bridges the DGX path planner to the comma body's existing motor control.
Must run on the comma body alongside the agent.

On a dev machine it runs in **dry-run** mode (logs commands to stdout).

Usage
-----
    uv run body-driver                          # dry-run (no hardware)
    uv run body-driver --dgx tcp/192.168.1.100:7447  # real comma body
"""

from __future__ import annotations

import argparse
import json
import time

import zenoh

from rerun_prompt_da.motor_controller import VELOCITY_TOPIC

# Mapping from differential-drive (linear, angular) to joystick (accel, steer).
# The comma body testJoystick: axes[0]=accel (0-0.6 fwd), axes[1]=steer (-1 to 1)
MAX_LINEAR = 0.3    # m/s from pure pursuit
MAX_ACCEL = 0.4     # joystick accel range (conservative)
MAX_STEER = 1.0

DEFAULT_DGX_ENDPOINT = "tcp/100.94.67.9:7447"


def velocity_to_joystick(linear: float, angular: float) -> tuple[float, float]:
    """Convert (linear m/s, angular rad/s) to (accel, steer) axes."""
    accel = (linear / MAX_LINEAR) * MAX_ACCEL if MAX_LINEAR > 0 else 0.0
    accel = max(-MAX_ACCEL, min(MAX_ACCEL, accel))

    # Positive angular = turning left in diff-drive, but steer axis:
    # negative = left, positive = right. So negate.
    steer = -(angular / 1.2) * MAX_STEER
    steer = max(-MAX_STEER, min(MAX_STEER, steer))

    return accel, steer


class DryRunDriver:
    """Prints velocity commands instead of sending them to hardware."""

    def __init__(self):
        self._last_print = 0.0

    def send(self, linear: float, angular: float):
        accel, steer = velocity_to_joystick(linear, angular)
        now = time.monotonic()
        if now - self._last_print > 0.25:
            print(f"[dry-run] linear={linear:+.3f} angular={angular:+.3f} -> accel={accel:+.3f} steer={steer:+.3f}")
            self._last_print = now


class CommaBodyDriver:
    """Send testJoystick cereal messages to drive the comma body."""

    def __init__(self):
        import cereal.messaging as messaging
        from openpilot.common.params import Params
        Params().put_bool('JoystickDebugMode', True)
        self._pm = messaging.PubMaster(['testJoystick'])
        print("CommaBodyDriver: JoystickDebugMode enabled, publishing testJoystick")

    def send(self, linear: float, angular: float):
        import cereal.messaging as messaging
        accel, steer = velocity_to_joystick(linear, angular)

        joy_msg = messaging.new_message('testJoystick')
        joy_msg.valid = True
        joy_msg.testJoystick.axes = [accel, steer]
        self._pm.send('testJoystick', joy_msg)


def main():
    parser = argparse.ArgumentParser(description="DGX velocity -> comma testJoystick bridge")
    parser.add_argument(
        "--dgx", type=str, default=None, metavar="ENDPOINT",
        help=f"DGX Zenoh endpoint (default: {DEFAULT_DGX_ENDPOINT})",
    )
    parser.add_argument(
        "--connect", type=str, default=None,
        help="Zenoh router endpoint (e.g. tcp/localhost:7447)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands instead of sending to hardware",
    )
    args = parser.parse_args()

    # Select driver backend.
    if args.dry_run:
        print("Running in dry-run mode")
        driver = DryRunDriver()
    else:
        driver = CommaBodyDriver()

    # Zenoh setup.
    conf = zenoh.Config()
    endpoint = args.dgx or args.connect or DEFAULT_DGX_ENDPOINT
    conf.insert_json5("connect/endpoints", f'["{endpoint}"]')
    session = zenoh.open(conf)

    last_cmd_time = 0.0

    def _on_velocity(sample):
        nonlocal last_cmd_time
        try:
            msg = json.loads(sample.payload.to_bytes().decode())
            linear = float(msg.get("linear", 0.0))
            angular = float(msg.get("angular", 0.0))
            driver.send(linear, angular)
            last_cmd_time = time.monotonic()
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass

    sub = session.declare_subscriber(VELOCITY_TOPIC, _on_velocity)
    print(f"Body driver: subscribed to '{VELOCITY_TOPIC}'")
    print(f"Connected to DGX at {endpoint}")
    print("Forwarding velocity commands to testJoystick. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
            # Safety: if no commands for 1s, send zero
            if time.monotonic() - last_cmd_time > 1.0 and last_cmd_time > 0:
                driver.send(0.0, 0.0)
    except KeyboardInterrupt:
        print("\nBody driver stopped.")
    finally:
        sub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
