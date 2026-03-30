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
import os
import signal
import subprocess
import time

import zenoh

from rerun_prompt_da.motor_controller import VELOCITY_TOPIC

# Mapping from differential-drive (linear, angular) to joystick (accel, steer).
# The comma body testJoystick: axes[0]=accel (0-0.6 fwd), axes[1]=steer (-1 to 1)
MAX_LINEAR = 0.9    # m/s (3x speed)
MAX_ACCEL = 1.2     # joystick accel range (3x speed)
MAX_STEER = 1.0

DEFAULT_DGX_ENDPOINT = "tcp/100.94.67.9:7447"
TOPIC = "testJoystick"


def velocity_to_joystick(linear: float, angular: float) -> tuple[float, float]:
    """Convert (linear m/s, angular rad/s) to (accel, steer) axes."""
    accel = (linear / MAX_LINEAR) * MAX_ACCEL if MAX_LINEAR > 0 else 0.0
    accel = max(-MAX_ACCEL, min(MAX_ACCEL, accel))

    # Positive angular = turning left in diff-drive, but steer axis:
    # negative = left, positive = right. So negate.
    steer = -(angular / 3.6) * MAX_STEER
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


def _kill_competing_publishers():
    """Kill other body_driver / joystick processes and remove stale IPC sockets."""
    my_pid = os.getpid()
    for pattern in ("body_driver", "joystick.py"):
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True,
            )
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                if pid != my_pid:
                    print(f"Killing competing process ({pattern}) pid={pid}")
                    os.kill(pid, signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    # msgq uses shared-memory files for IPC; remove the stale binding
    for path in (f"/dev/shm/{TOPIC}", f"/tmp/{TOPIC}"):
        try:
            if os.path.exists(path):
                os.unlink(path)
                print(f"Removed stale IPC socket: {path}")
        except OSError:
            pass

    time.sleep(0.3)


class CommaBodyDriver:
    """Send testJoystick cereal messages to drive the comma body."""

    def __init__(self):
        import cereal.messaging as messaging
        from openpilot.common.params import Params

        _kill_competing_publishers()

        Params().put_bool('JoystickDebugMode', True)
        self._messaging = messaging
        self._pm = messaging.PubMaster([TOPIC])
        self._send_errors = 0
        print(f"CommaBodyDriver: JoystickDebugMode enabled, publishing {TOPIC}")

    def _recreate_publisher(self):
        """Tear down and recreate the PubMaster after a socket conflict."""
        print(f"Recreating PubMaster for {TOPIC}...")
        _kill_competing_publishers()
        try:
            del self._pm
        except AttributeError:
            pass
        self._pm = self._messaging.PubMaster([TOPIC])
        self._send_errors = 0

    def send(self, linear: float, angular: float):
        accel, steer = velocity_to_joystick(linear, angular)

        joy_msg = self._messaging.new_message(TOPIC)
        joy_msg.valid = True
        joy_msg.testJoystick.axes = [accel, steer]
        try:
            self._pm.send(TOPIC, joy_msg)
            self._send_errors = 0
        except Exception as exc:
            self._send_errors += 1
            if self._send_errors <= 3:
                print(f"Send error ({self._send_errors}/3): {exc}")
                self._recreate_publisher()
            elif self._send_errors == 4:
                print("Persistent publisher conflict — suppressing further errors. "
                      "Make sure no other process is publishing to testJoystick.")


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
        except Exception as exc:
            print(f"callback error: {exc}")

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
