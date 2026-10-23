"""Flexor node and shared ROS runtime; node boundaries carry timestamped spikes."""

import gc
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int64MultiArray, String

from recipe import SIDES, Controller, finish_measurement, start_measurement, window_steps


class RuntimeNode(Node):
    """Absolute release schedule and buffered diagnostics, exported after control."""

    PROFILE_STAGES = ("callback",)

    def __init__(self, name):
        super().__init__(name)
        self.dt_ms = float(self.declare_parameter("dt_ms", 0.1).value)
        self.timer_ms = float(self.declare_parameter("timer_ms", 1.0).value)
        self.substeps = window_steps(self.dt_ms, self.timer_ms)
        self.duration = float(self.declare_parameter("duration", 8.0).value)
        self.neurons = int(self.declare_parameter("neurons", 1).value)
        self.max_lag_ms = float(self.declare_parameter("max_lag_ms", 20.0).value)
        self.profile_timing = bool(self.declare_parameter("profile_timing", False).value)
        self.declare_parameter("start_ns", 0)
        if self.duration <= 0 or self.neurons < 1 or self.max_lag_ms <= 0:
            raise ValueError("Invalid duration, population size or maximum lag")
        self.cpu_affinity = sorted(os.sched_getaffinity(0))
        self.steps = 0
        self.updates = 0
        self.capacity = math.ceil(self.duration * 1000 / self.timer_ms)
        # Fixed numeric storage avoids per-window dictionaries and JSON in control.
        self.timing = np.zeros((self.capacity, 5))
        self.stamps = np.zeros(self.capacity, dtype=np.int64)
        self.stage_timing = (
            np.zeros((self.capacity, len(self.PROFILE_STAGES), 2)) if self.profile_timing else None
        )
        self.clock_probe = None
        if self.profile_timing:
            probe = np.zeros((1000, 2))
            for row in probe:
                measured = start_measurement()
                finish_measurement(measured, row)
            self.clock_probe = dict(
                samples=len(probe),
                empty_span_wall_ms_median=float(np.median(probe[:, 0])),
                empty_span_wall_ms_p95=float(np.percentile(probe[:, 0], 95)),
                empty_span_cpu_ms_median=float(np.median(probe[:, 1])),
            )
        self.deliveries = []
        self.sequences = {}
        self.received_sequences = {}
        self.diagnostics = self.create_publisher(String, "/diagnostics/" + name, 100)
        self.finished = False
        self.stop_reason = "not_started"
        self.clock = self.get_clock()
        self.period_ns = round(self.timer_ms * 1e6)

    def execute(self):
        """Service ROS while waiting for fixed releases; never shift the origin."""
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        start_ns = 0
        while rclpy.ok() and not start_ns:
            executor.spin_once(timeout_sec=0.01)
            start_ns = int(self.get_parameter("start_ns").value)
        if not rclpy.ok():
            return
        # Map the common ROS wall-clock epoch to a local monotonic deadline once.
        before = time.perf_counter_ns()
        wall = self.clock.now().nanoseconds
        after = time.perf_counter_ns()
        origin = (before + after) // 2 + start_ns - wall
        end = origin + round(self.duration * 1e9)
        self.stop_reason = "completed"
        # The bounded run buffers diagnostics; collect cyclic garbage outside control.
        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        while rclpy.ok() and self.updates < self.capacity:
            release = origin + self.updates * self.period_ns
            now = time.perf_counter_ns()
            while now < release and rclpy.ok():
                # Service ROS at least once per period while armed, including startup.
                executor.spin_once(timeout_sec=min((release - now) / 1e9, self.timer_ms / 1000))
                now = time.perf_counter_ns()
            if not rclpy.ok():
                self.stop_reason = "interrupted"
                break
            if now >= end:
                self.stop_reason = "wall_window_ended"
                break
            if now - release > self.max_lag_ms * 1e6:
                self.stop_reason = "excessive_lag"
                break
            # If late, preserve all model windows and the original deadlines.
            # Catch up immediately, but service queued spikes between late windows.
            if now - release >= self.period_ns:
                executor.spin_once(timeout_sec=0)
            measured = start_measurement() if self.profile_timing else None
            started = measured[0] if measured is not None else time.perf_counter_ns()
            wall_start = self.clock.now().nanoseconds
            index = self.updates
            compute_ms = self.tick()
            self.steps += self.substeps
            self.stamps[index] = self.clock.now().nanoseconds
            self.timing[index, 0] = (wall_start - start_ns - index * self.period_ns) / 1e6
            self.timing[index, 1] = compute_ms
            self.timing[index, 2] = self.steps * self.dt_ms / 1000
            # Include model work, spike publication and writes to diagnostic buffers.
            if measured is not None:
                finish_measurement(measured, self.stage_timing[index, -1])
                self.timing[index, 3] = self.stage_timing[index, -1, 0]
            else:
                self.timing[index, 3] = (time.perf_counter_ns() - started) / 1e6
            self.timing[index, 4] = self.timing[index, 0] + self.timing[index, 3] - self.timer_ms
            self.updates += 1
        # Export only after the shared control interval, including when stopping early.
        while rclpy.ok() and time.perf_counter_ns() < end:
            executor.spin_once(timeout_sec=max(0, (end - time.perf_counter_ns()) / 1e9))
        if gc_was_enabled:
            gc.enable()
        self.flush()
        # Keep publishers alive while the recorder drains reliable diagnostics.
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def emit(self, publisher, neuron_id=0, model_time_ms=0.0):
        topic = publisher.topic_name
        seq = self.sequences.get(topic, 0)
        self.sequences[topic] = seq + 1
        publisher.publish(
            Int64MultiArray(
                data=[seq, self.clock.now().nanoseconds, int(neuron_id), round(model_time_ms * 1e6)]
            )
        )

    def receive(self, msg, topic):
        if len(msg.data) != 4:
            raise ValueError(
                "Spike payload must be [sequence, publication_ns, neuron_id, source_model_time_ns]"
            )
        sequence, stamp, _neuron_id, model_time_ns = msg.data
        now = self.clock.now().nanoseconds
        previous = self.received_sequences.get(topic, -1)
        self.deliveries.append(
            (now, stamp, topic, model_time_ns, (now - stamp) / 1e6, max(0, sequence - previous - 1))
        )
        if sequence <= previous:
            return False
        self.received_sequences[topic] = sequence
        return True

    def diagnostic_values(self, index):
        raise NotImplementedError

    def flush(self):
        if self.finished:
            return
        records = []
        for index in range(self.updates):
            lag, compute, model_time, callback, lateness = self.timing[index]
            records.append(
                dict(
                    kind="tick",
                    stamp_ns=int(self.stamps[index]),
                    model_time_s=model_time,
                    wall_lag_ms=lag,
                    compute_ms=compute,
                    callback_ms=callback,
                    deadline_lateness_ms=lateness,
                    **self.diagnostic_values(index),
                    **(
                        {"stage_timing_ms": self.stage_timing[index].tolist()}
                        if self.profile_timing
                        else {}
                    ),
                    **(
                        {"profile_stages": self.PROFILE_STAGES, "clock_probe": self.clock_probe}
                        if self.profile_timing and index == 0
                        else {}
                    ),
                    **({"cpu_affinity": self.cpu_affinity} if index == 0 else {}),
                )
            )
        records.extend(
            dict(
                kind="delivery",
                stamp_ns=now,
                sent_ns=stamp,
                topic=topic,
                source_model_time_ns=model,
                age_ms=age,
                missing=missing,
            )
            for now, stamp, topic, model, age, missing in self.deliveries
        )
        records.append(dict(kind="finished", stop_reason=self.stop_reason))
        for offset in range(0, len(records), 100):
            self.diagnostics.publish(String(data=json.dumps(records[offset : offset + 100])))
        self.finished = True


class ControllerNode(RuntimeNode):
    PROFILE_STAGES = (
        "input_preparation",
        "snnlab_executor",
        "output_extraction",
        "spike_publication",
        "callback",
    )

    def __init__(self, side):
        super().__init__(side)
        self.side = side
        self.model = Controller(
            side,
            self.dt_ms,
            self.neurons,
            warm_steps=self.substeps,
            profile_timing=self.profile_timing,
        )
        self.voltage = np.zeros((self.capacity, self.substeps, 2))
        self.spikes = np.zeros((self.capacity, self.substeps, 2), dtype=int)
        self.pending_sensory, self.pending_peer = 0, 0
        self.publisher = self.create_publisher(Int64MultiArray, "/motor/" + side, 1000)
        self.sensory_topic = "/sensory/" + side
        self.peer_topic = "/motor/" + SIDES[1 - SIDES.index(side)]
        self.subscriptions_ = [
            self.create_subscription(Int64MultiArray, self.sensory_topic, self.sensory, 1000),
            self.create_subscription(Int64MultiArray, self.peer_topic, self.peer, 1000),
        ]
        self.get_logger().info(
            f"Ready: {2 * self.neurons} neurons; CPU {self.cpu_affinity}; {self.model.info}"
        )

    def sensory(self, msg):
        if self.receive(msg, self.sensory_topic):
            self.pending_sensory += 1

    def peer(self, msg):
        if self.receive(msg, self.peer_topic):
            self.pending_peer += 1

    def tick(self):
        tick = time.perf_counter_ns()
        voltage, spikes, events = self.model.window(
            self.substeps, self.pending_sensory, self.pending_peer
        )
        self.pending_sensory, self.pending_peer = 0, 0
        measured = start_measurement() if self.profile_timing else None
        for offset, neuron_id in events:
            self.emit(self.publisher, neuron_id, (self.steps + offset + 1) * self.dt_ms)
        if measured is not None:
            finish_measurement(measured, self.stage_timing[self.updates, 3])
            self.stage_timing[self.updates, :3] = self.model.stage_timing
        compute_ms = (time.perf_counter_ns() - tick) / 1e6
        self.voltage[self.updates] = voltage
        self.spikes[self.updates] = spikes
        return compute_ms

    def diagnostic_values(self, index):
        return dict(
            voltage=self.voltage[index, -1].tolist(),
            spikes=self.spikes[index].sum(axis=0).tolist(),
            substep_voltage=self.voltage[index].tolist(),
            substep_spikes=self.spikes[index].tolist(),
            **({"model_info": self.model.info} if index == 0 else {}),
        )


def run(side="flexor"):
    rclpy.init()
    node = ControllerNode(side)
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run("flexor")
