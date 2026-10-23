"""Mechanical plant ROS node; all controller-facing messages are spikes."""

import time

import numpy as np
import rclpy
from std_msgs.msg import Int64MultiArray

from node_flexor import RuntimeNode
from recipe import SIDES, Joint, finish_measurement, start_measurement


class PlantNode(RuntimeNode):
    PROFILE_STAGES = ("input_preparation", "plant_dynamics", "spike_publication", "callback")

    def __init__(self):
        super().__init__("plant")
        self.plant = Joint()
        self.pending = np.zeros(2)
        self.publishers_ = [
            self.create_publisher(Int64MultiArray, "/sensory/" + s, 1000) for s in SIDES
        ]
        self.subscriptions_ = [
            self.create_subscription(
                Int64MultiArray,
                "/motor/" + s,
                lambda msg, i=i: self.motor(i, msg),
                1000,
            )
            for i, s in enumerate(SIDES)
        ]
        self.state_buffer = np.zeros((self.capacity, 6))
        self.motor_buffer = np.zeros(2)
        self.get_logger().info(f"Ready: mechanical joint, CPU {self.cpu_affinity}, no neurons")

    def motor(self, i, msg):
        if self.receive(msg, "/motor/" + SIDES[i]):
            self.pending[i] += 1

    def tick(self):
        tick = time.perf_counter_ns()
        measured = start_measurement() if self.profile_timing else None
        events = []
        np.divide(self.pending, self.neurons, out=self.motor_buffer)
        self.pending.fill(0)
        if measured is not None:
            finish_measurement(measured, self.stage_timing[self.updates, 0])
        measured = start_measurement() if self.profile_timing else None
        for substep in range(self.substeps):
            model_ms = (self.steps + substep) * self.dt_ms
            sensory = self.plant.encode(model_ms / 1000)
            for i, count in enumerate(sensory):
                if count:
                    events.append((i, model_ms))
            # Received motor events enter once, before the first decay/update.
            torque = self.plant.advance(self.motor_buffer, self.dt_ms / 1000)
            self.motor_buffer.fill(0)
        if measured is not None:
            finish_measurement(measured, self.stage_timing[self.updates, 1])
        measured = start_measurement() if self.profile_timing else None
        for i, model_ms in events:
            self.emit(self.publishers_[i], model_time_ms=model_ms)
        if measured is not None:
            finish_measurement(measured, self.stage_timing[self.updates, 2])
        compute_ms = (time.perf_counter_ns() - tick) / 1e6
        self.state_buffer[self.updates] = (
            self.plant.angle,
            self.plant.velocity,
            *self.plant.activation,
            torque,
            self.plant.limit_hits,
        )
        return compute_ms

    def diagnostic_values(self, index):
        angle, velocity, flexor, extensor, torque, hits = self.state_buffer[index]
        return dict(
            angle=angle,
            velocity=velocity,
            activation=[flexor, extensor],
            torque=torque,
            limit_hits=int(hits),
        )


def main():
    rclpy.init()
    node = PlantNode()
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
    main()
