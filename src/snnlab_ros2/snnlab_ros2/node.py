"""custom_ring-compatible ROS topics backed by a persistent SNNLab worker."""

import json
import math
import os
from pathlib import Path
import select
import subprocess
import time

from builtin_interfaces.msg import Time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class SNNLabNeuron(Node):
    def __init__(self):
        super().__init__("snnlab_neuron")
        defaults = {"topic_prefix": "/custom_ring", "neuron_id": "n1",
                    "predecessor_id": "n3", "wall_timer_s": 0.01,
                    "tonic_current": 0.0}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        neuron = str(self.get_parameter("neuron_id").value)
        predecessor = str(self.get_parameter("predecessor_id").value)
        self.period = float(self.get_parameter("wall_timer_s").value)
        self.current = float(self.get_parameter("tonic_current").value)
        if not math.isfinite(self.period) or self.period <= 0:
            raise ValueError("wall_timer_s must be finite and positive")
        if not math.isfinite(self.current) or self.current < 0:
            raise ValueError("tonic_current must be finite and nonnegative")
        self.pending = []
        self.dropped_events = 0
        self.tick_id = 0
        self.last_tick_ns = None
        env = os.environ.copy()
        # Share only our adapter package with the isolated interpreter, without
        # inheriting Humble's Python extension paths.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env.pop("PYTHONHOME", None)
        self.worker = subprocess.Popen(
            [os.environ.get("SNNLAB_PYTHON", "/opt/snnlab-venv/bin/python"),
             "-u", "-m", "snnlab_ros2.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env=env,
        )
        try:
            self.metadata = self._read_worker(timeout=60.0)
            if not self.metadata.get("ready"):
                raise RuntimeError("SNNLab worker did not report readiness")
            self.event_pub = self.create_publisher(Time, f"{prefix}/{neuron}/events", 10)
            self.spike_pub = self.create_publisher(Time, f"{prefix}/{neuron}/spikes", 10)
            self.voltage_pub = self.create_publisher(Float32, f"{prefix}/{neuron}/voltage", 10)
            self.timing_pub = self.create_publisher(String, f"{prefix}/{neuron}/timing", 10)
            self.event_sub = self.create_subscription(
                Time, f"{prefix}/{predecessor}/events", self._event, 10)
            self.current_sub = self.create_subscription(
                Float32, f"{prefix}/tonic_current", self._current, 10)
            self.timer = self.create_timer(self.period, self._tick)
        except BaseException:
            self._stop_worker()
            raise
        self.get_logger().info(
            f"Ready: {prefix}/{predecessor}/events -> SNNLab -> {prefix}/{neuron}; "
            f"{self.metadata['steps']} steps/tick, dt={self.metadata['dt_ms']} ms")

    def _read_worker(self, timeout=5.0):
        readable, _, _ = select.select([self.worker.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("SNNLab worker response timed out")
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError(f"SNNLab worker exited (status={self.worker.poll()})")
        return json.loads(line)

    def _event(self, msg):
        if len(self.pending) >= 1000:
            self.dropped_events += 1
            return
        self.pending.append({"stamp_ns": msg.sec * 1_000_000_000 + msg.nanosec,
                             "received_ns": self.get_clock().now().nanoseconds})

    def _current(self, msg):
        value = float(msg.data)
        if math.isfinite(value) and value >= 0:
            self.current = value
        else:
            self.get_logger().warning("Ignoring nonfinite or negative tonic current")

    def _tick(self):
        started = time.perf_counter_ns()
        interval_ms = (None if self.last_tick_ns is None else
                       (started - self.last_tick_ns) / 1e6)
        self.last_tick_ns = started
        events, self.pending = self.pending, []
        dispatched_ns = self.get_clock().now().nanoseconds
        self.worker.stdin.write(json.dumps({"current_na": self.current,
                                             "event_count": len(events)}) + "\n")
        self.worker.stdin.flush()
        result = self._read_worker()
        self.voltage_pub.publish(Float32(data=result["voltage_mv"]))
        for _offset in result["spike_offsets"]:
            # Match custom_ring's publication-time stamps, not simulation time.
            stamp = self.get_clock().now().to_msg()
            self.event_pub.publish(stamp)
            self.spike_pub.publish(stamp)
        adapter_ms = (time.perf_counter_ns() - started) / 1e6
        self.tick_id += 1
        self.timing_pub.publish(String(data=json.dumps({
            "tick_id": self.tick_id,
            "completed_steps": result["completed_steps"],
            "sim_time_ms": result["completed_steps"] * self.metadata["dt_ms"],
            "current_na": self.current,
            "inhibitory_peak_na": result["inhibitory_peak_na"],
            "compute_ms": result["compute_ms"],
            "adapter_ms": adapter_ms,
            "timer_interval_ms": interval_ms,
            "overrun": adapter_ms > self.period * 1000,
            "dispatched_ns": dispatched_ns,
            "published_ns": self.get_clock().now().nanoseconds,
            "events": events,
            "dropped_events": self.dropped_events,
        }, allow_nan=False)))

    def _stop_worker(self):
        if self.worker.stdin:
            try:
                self.worker.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        if self.worker.stdout:
            self.worker.stdout.close()

    def destroy_node(self):
        self._stop_worker()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SNNLabNeuron()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
