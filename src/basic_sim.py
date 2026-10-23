"""Standalone reference: two SNNLab controllers and a mechanical joint.

Run: uv run python src/basic_sim.py
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import snnlab

from present import save
from recipe import DT_CHOICES, Controller, Joint


def simulate(duration=8.0, dt_ms=0.1, feedback=True, neurons=1):
    dt = dt_ms / 1000
    steps = round(duration / dt)
    setup_start = time.perf_counter()
    controllers = [Controller(side, dt_ms, neurons) for side in ("flexor", "extensor")]
    plant = Joint()
    setup_seconds = time.perf_counter() - setup_start
    values = dict(
        time_s=(np.arange(steps) + 1) * dt,
        sensory=np.zeros((steps, 2)),
        voltage=np.zeros((steps, 4)),
        spikes=np.zeros((steps, 4)),
        activation=np.zeros((steps, 2)),
        angle=np.zeros(steps),
        velocity=np.zeros(steps),
        torque=np.zeros(steps),
        controller_ms=np.zeros(steps),
        plant_ms=np.zeros(steps),
        loop_ms=np.zeros(steps),
    )
    peers = [0, 0]
    start = time.perf_counter()
    for k in range(steps):
        tick = time.perf_counter_ns()
        sensory = plant.encode(k * dt, feedback)
        neural_start = time.perf_counter_ns()
        for i, controller in enumerate(controllers):
            voltage, spikes, _ = controller.step(sensory[i], peers[i])
            values["voltage"][k, 2 * i : 2 * i + 2] = voltage
            values["spikes"][k, 2 * i : 2 * i + 2] = spikes
        peers = [int(values["spikes"][k, 2]), int(values["spikes"][k, 0])]
        neural_end = time.perf_counter_ns()
        plant_start = time.perf_counter_ns()
        values["torque"][k] = plant.advance(values["spikes"][k, [0, 2]] / neurons, dt)
        plant_end = time.perf_counter_ns()
        values["sensory"][k] = sensory
        values["activation"][k] = plant.activation
        values["angle"][k], values["velocity"][k] = plant.angle, plant.velocity
        values["controller_ms"][k] = (neural_end - neural_start) / 1e6
        values["plant_ms"][k] = (plant_end - plant_start) / 1e6
        values["loop_ms"][k] = (time.perf_counter_ns() - tick) / 1e6
    summary = summarize(values, dt_ms, time.perf_counter() - start, setup_seconds, plant, feedback)
    summary.update(
        mode="basic",
        neurons_per_population=neurons,
        total_neurons=4 * neurons,
        controller_execution={c.side: c.info for c in controllers},
    )
    if not all(np.isfinite(v).all() for v in values.values()):
        raise RuntimeError("Non-finite simulation state")
    return values, summary


def summarize(values, dt_ms, wall_seconds, setup_seconds, plant, feedback):
    # Count changes of motor dominance in non-overlapping 100 ms windows.
    bin_steps = max(1, round(100 / dt_ms))
    counts = np.array(
        [
            values["spikes"][start : start + bin_steps, [0, 2]].sum(axis=0)
            for start in range(0, len(values["time_s"]), bin_steps)
        ]
    )
    dominance = np.sign(counts[:, 0] - counts[:, 1])
    active = dominance[dominance != 0]
    handoffs = int(np.count_nonzero(np.diff(active)))
    moving = np.sign(values["velocity"][np.abs(values["velocity"]) > 0.05])
    reversals = int(np.count_nonzero(np.diff(moving)))
    return {
        "snnlab_version": snnlab.__version__,
        "feedback": feedback,
        "dt_ms": dt_ms,
        "simulated_seconds": len(values["time_s"]) * dt_ms / 1000,
        "setup_seconds": setup_seconds,
        "loop_wall_seconds": wall_seconds,
        "simulated_seconds_per_wall_second": len(values["time_s"]) * dt_ms / 1000 / wall_seconds,
        "neuron_order": [
            "flexor motor",
            "flexor slow",
            "extensor motor",
            "extensor slow",
        ],
        "spikes": values["spikes"].sum(axis=0).astype(int).tolist(),
        "sensory_spikes": values["sensory"].sum(axis=0).astype(int).tolist(),
        "motor_dominance_handoffs": handoffs,
        "joint_direction_reversals": reversals,
        "angle_range_rad": [float(values["angle"].min()), float(values["angle"].max())],
        "mechanical_stop_hits": plant.limit_hits,
        "loop_compute_ms": {
            "median": float(np.median(values["loop_ms"])),
            "p95": float(np.percentile(values["loop_ms"], 95)),
            "maximum": float(values["loop_ms"].max()),
        },
        "over_budget_steps": int((values["loop_ms"] > dt_ms).sum()),
        "timing_scope": "Unpaced CPU simulation: measured compute versus simulated step budget; no ROS or hard real-time guarantee",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt-ms", type=float, choices=DT_CHOICES, default=0.1)
    parser.add_argument("--neurons", type=int, default=1, help="Neurons per population; total = 4N")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/basic",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration * 1000 < args.dt_ms or args.neurons < 1:
        parser.error("Require a positive duration of at least one timestep, and neurons >= 1")
    values, summary = simulate(args.duration, args.dt_ms, not args.no_feedback, args.neurons)
    save(values, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
