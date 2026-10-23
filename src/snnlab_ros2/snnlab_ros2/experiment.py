"""Finite DDS integration test, latency measurements and headless plot export."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from builtin_interfaces.msg import Time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class Probe(Node):
    def __init__(self):
        super().__init__("snnlab_experiment")
        self.origin_ns = time.perf_counter_ns()
        self.phase = "settling"
        self.voltages = []
        self.spikes = []
        self.events = []
        self.timings = []
        self.sent = {}
        self.acks = []
        self.current_pub = self.create_publisher(Float32, "/custom_ring/tonic_current", 10)
        self.input_pub = self.create_publisher(Time, "/custom_ring/n3/events", 10)
        self.subscriptions_kept = [
            self.create_subscription(Float32, "/custom_ring/n1/voltage", self._voltage, 10),
            self.create_subscription(Time, "/custom_ring/n1/spikes", self._spike, 10),
            self.create_subscription(Time, "/custom_ring/n1/events", self._event, 10),
            self.create_subscription(String, "/custom_ring/n1/timing", self._timing, 10),
        ]

    def elapsed(self):
        return (time.perf_counter_ns() - self.origin_ns) / 1e9

    def _voltage(self, msg):
        self.voltages.append({"t_s": self.elapsed(), "phase": self.phase, "mv": msg.data})

    def _spike(self, msg):
        self.spikes.append({"t_s": self.elapsed(), "phase": self.phase,
                            "stamp_ns": msg.sec * 1_000_000_000 + msg.nanosec})

    def _event(self, msg):
        self.events.append({"t_s": self.elapsed(), "phase": self.phase,
                            "stamp_ns": msg.sec * 1_000_000_000 + msg.nanosec})

    def _timing(self, msg):
        received_perf_ns = time.perf_counter_ns()
        received_ros_ns = self.get_clock().now().nanoseconds
        row = json.loads(msg.data)
        row.update(t_s=self.elapsed(), phase=self.phase,
                   telemetry_delivery_ms=(received_ros_ns - row["published_ns"]) / 1e6)
        self.timings.append(row)
        for event in row["events"]:
            sent = self.sent.get(event["stamp_ns"])
            if sent is not None:
                self.acks.append({
                    "stamp_ns": event["stamp_ns"],
                    "input_delivery_ms": (event["received_ns"] - event["stamp_ns"]) / 1e6,
                    "application_dispatch_ms": (row["dispatched_ns"] - event["stamp_ns"]) / 1e6,
                    "ack_roundtrip_ms": (received_perf_ns - sent["perf_ns"]) / 1e6,
                })

    def send_event(self):
        perf_ns = time.perf_counter_ns()
        stamp = self.get_clock().now().to_msg()
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        self.sent[stamp_ns] = {"perf_ns": perf_ns, "t_s": self.elapsed()}
        self.input_pub.publish(stamp)

    def connected(self):
        return (self.current_pub.get_subscription_count() > 0
                and self.input_pub.get_subscription_count() > 0
                and all(self.count_publishers(f"/custom_ring/n1/{suffix}") > 0
                        for suffix in ("voltage", "events", "spikes", "timing"))
                and bool(self.timings))


def spin_for(probe, child, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Neuron process exited with {child.returncode}; see node.log")
        rclpy.spin_once(probe, timeout_sec=min(0.005, max(0, deadline - time.monotonic())))


def stats(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def evaluate(probe, phases):
    actual = [row for row in probe.timings if row["phase"] in phases]
    recovery = phases.get("recovery")
    quiet_end = float("inf") if recovery is None else sum(recovery) / 2
    observed = {row["stamp_ns"] for row in probe.acks}
    checks = {
        "voltage_received_each_phase": len(phases) == 4 and all(
            any(row["phase"] == phase for row in probe.voltages) for phase in phases),
        "finite_voltage": bool(probe.voltages) and all(
            math.isfinite(row["mv"]) for row in probe.voltages),
        "driven_neuron_spikes": any(row["phase"] == "driven" for row in probe.spikes),
        "baseline_silent": "baseline" in phases and not any(
            row["phase"] == "baseline" for row in probe.spikes),
        "recovery_becomes_silent": recovery is not None and not any(
            row["phase"] == "recovery" and row["t_s"] > quiet_end for row in probe.spikes),
        "all_input_events_applied": bool(probe.sent) and observed == set(probe.sent)
            and len(probe.acks) == len(probe.sent),
        "inhibitory_current_observed": any(row["inhibitory_peak_na"] > 0 for row in actual),
        "state_steps_continue": len(actual) > 1 and all(
            right["completed_steps"] > left["completed_steps"]
            for left, right in zip(actual, actual[1:])),
        "events_match_spikes": bool(probe.spikes) and Counter(
            row["stamp_ns"] for row in probe.events) == Counter(
            row["stamp_ns"] for row in probe.spikes),
        "no_application_queue_drops": bool(actual) and all(
            row["dropped_events"] == 0 for row in actual),
    }
    metrics = {name: stats([row[name] for row in actual if row[name] is not None])
               for name in ("compute_ms", "adapter_ms", "timer_interval_ms", "telemetry_delivery_ms")}
    metrics.update({name: stats([row[name] for row in probe.acks]) for name in
                    ("input_delivery_ms", "application_dispatch_ms", "ack_roundtrip_ms")})
    metrics["adapter_overruns"] = sum(row["overrun"] for row in actual)
    metrics["spikes_by_phase"] = dict(Counter(row["phase"] for row in probe.spikes))
    return checks, metrics


def plot(probe, phases, output):
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    colours = {"baseline": "#ccd5df", "driven": "#b9e4c9",
               "inhibited": "#edc3b6", "recovery": "#c9c8e8"}
    for phase, (start, end) in phases.items():
        for axis in axes[:3]:
            axis.axvspan(start, end, color=colours[phase], alpha=0.3)
        axes[0].text((start + end) / 2, 1.02, phase, ha="center",
                     transform=axes[0].get_xaxis_transform())
    axes[0].plot([r["t_s"] for r in probe.voltages], [r["mv"] for r in probe.voltages], lw=0.8)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("SNNLab / ROS 2 Humble integration experiment", pad=30)
    axes[1].scatter([r["t_s"] for r in probe.spikes], [1] * len(probe.spikes), marker="|", label="LIF spikes")
    axes[1].scatter([r["t_s"] for r in probe.sent.values()], [0] * len(probe.sent), marker="|", label="Input events")
    axes[1].set_yticks([0, 1], ["Input", "Output"])
    axes[1].set_ylim(-0.5, 1.5)
    for name, label in (("compute_ms", "SNNLab compute"), ("adapter_ms", "Adapter incl. IPC")):
        axes[2].plot([r["t_s"] for r in probe.timings], [r[name] for r in probe.timings], label=label, lw=0.8)
    axes[2].axhline(10, color="black", ls="--", lw=0.8, label="10 ms timer period")
    axes[2].set_ylabel("Duration (ms)")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Elapsed wall time (s)")
    for name, label in (("application_dispatch_ms", "Input → worker dispatch"),
                        ("ack_roundtrip_ms", "Input → telemetry received")):
        values = [r[name] for r in probe.acks]
        if values:
            axes[3].hist(values, bins=min(15, len(values)), alpha=0.55, label=label)
    if probe.acks:
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "No acknowledged input events", ha="center", transform=axes[3].transAxes)
    axes[3].set_xlabel("Latency (ms)")
    axes[3].set_ylabel("Events")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "experiment.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/artifacts"))
    parser.add_argument("--phase-seconds", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    args = parser.parse_args()
    if not math.isfinite(args.phase_seconds) or args.phase_seconds < 1.0:
        parser.error("--phase-seconds must be finite and at least 1")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be finite and positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    probe = Probe()
    phases = {}
    error = None
    child = None
    try:
        with (output / "node.log").open("w") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "snnlab_ros2.node"],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            while not probe.connected():
                if time.monotonic() > deadline:
                    raise TimeoutError("ROS discovery / SNNLab startup timed out; see node.log")
                spin_for(probe, child, 0.05)
            spin_for(probe, child, 0.2)
            # Exclude startup/settling from the plot and measured experiment.
            probe.voltages.clear()
            probe.spikes.clear()
            probe.events.clear()
            probe.timings.clear()
            probe.origin_ns = time.perf_counter_ns()
            for phase, current, inject in (("baseline", 0.0, False), ("driven", 0.15, False),
                                           ("inhibited", 0.15, True), ("recovery", 0.0, False)):
                probe.phase = phase
                start = probe.elapsed()
                phases[phase] = [start, start + args.phase_seconds]
                end = time.monotonic() + args.phase_seconds
                next_publish = 0.0
                while time.monotonic() < end:
                    now = time.monotonic()
                    if now >= next_publish:
                        probe.current_pub.publish(Float32(data=current))
                        if inject:
                            probe.send_event()
                        next_publish = now + 0.05
                    spin_for(probe, child, min(0.005, max(0, end - now)))
                phases[phase][1] = probe.elapsed()
            probe.phase = "drain"
            spin_for(probe, child, 0.2)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        if child is not None:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        checks, metrics = evaluate(probe, phases)
        passed = error is None and len(phases) == 4 and all(checks.values())
        summary = {"passed": passed, "error": error, "checks": checks, "metrics": metrics,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "snnlab_ref": os.environ.get("SNNLAB_REF", "unknown"),
                   "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
                   "platform": platform.platform(), "python": platform.python_version(),
                   "phase_seconds": args.phase_seconds, "phases": phases,
                   "dt_ms": 1.0, "steps_per_tick": 10, "wall_timer_ms": 10.0,
                   "input_events_sent": len(probe.sent), "input_events_acked": len(probe.acks)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        traces = {"voltage": probe.voltages, "spikes": probe.spikes, "events": probe.events,
                  "timing": probe.timings, "sent": probe.sent, "acknowledgements": probe.acks}
        (output / "traces.json").write_text(json.dumps(traces, allow_nan=False) + "\n")
        plot(probe, phases, output)
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
