"""Run the standalone reference, then measure the three live ROS processes."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from present import save
from recipe import DT_CHOICES, SIDES, SNNLAB_REF, window_steps

NODES = ("flexor", "extensor", "plant")
TOPICS = tuple("/" + kind + "/" + side for kind in ("motor", "sensory") for side in SIDES)


def stats(values):
    data = np.asarray(values, dtype=float)
    if not len(data):
        return None
    return dict(
        n=len(data),
        median=float(np.median(data)),
        p95=float(np.percentile(data, 95)),
        maximum=float(np.max(data)),
    )


def profile_summary(values, ticks, period_ms):
    """Keep CPU work and non-CPU elapsed time distinct; neither establishes causality."""
    nodes = {}

    def span_stats(samples):
        if not len(samples):
            return None
        return {
            key: dict(stats(data), mean=float(np.mean(data)), minimum=float(np.min(data)))
            for key, data in (
                ("wall_ms", samples[:, 0]),
                ("thread_cpu_ms", samples[:, 1]),
                ("wall_minus_thread_cpu_ms", samples[:, 0] - samples[:, 1]),
            )
        }

    for name, rows in ticks.items():
        names = next((r["profile_stages"] for r in rows if "profile_stages" in r), [])
        if not names:
            continue
        samples = np.array([r["stage_timing_ms"] for r in rows])
        values[name + "_stage_wall_ms"] = samples[:, :, 0]
        values[name + "_stage_cpu_ms"] = samples[:, :, 1]
        late = values[name + "_lateness_ms"] > 0
        callback = samples[:, -1]
        lag = values[name + "_lag_ms"]
        already_late = lag >= period_ms
        cpu_exceeds_remaining = lag + callback[:, 1] > period_ms
        stages = {}
        for index, stage in enumerate(names):
            stages[stage] = dict(
                all_updates=span_stats(samples[:, index]),
                missed_deadlines=span_stats(samples[late, index]),
                callback_cpu_share_percent=float(
                    100 * samples[:, index, 1].sum() / callback[:, 1].sum()
                )
                if callback[:, 1].sum()
                else None,
            )
        other = callback - samples[:, :-1].sum(axis=1)
        stages["callback_other"] = dict(
            all_updates=span_stats(other),
            missed_deadlines=span_stats(other[late]),
            callback_cpu_share_percent=float(100 * other[:, 1].sum() / callback[:, 1].sum())
            if callback[:, 1].sum()
            else None,
        )
        nodes[name] = dict(
            stage_order=names,
            clock_probe=next(r["clock_probe"] for r in rows if "clock_probe" in r),
            stages=stages,
            missed_deadline_breakdown=dict(
                total=int(late.sum()),
                already_past_deadline_at_start=int((late & already_late).sum()),
                start_lag_plus_thread_cpu_exceeds_budget=int(
                    (late & ~already_late & cpu_exceeds_remaining).sum()
                ),
                elapsed_callback_gap_pushes_over_deadline=int(
                    (late & ~already_late & ~cpu_exceeds_remaining).sum()
                ),
            ),
        )
    return dict(
        enabled=bool(nodes),
        nodes=nodes,
        units="milliseconds",
        method="perf_counter_ns elapsed wall time; thread_time_ns CPU time of the calling thread",
        interpretation="Wall minus thread CPU is a signed estimate of blocking/descheduling plus clock-probe error. It excludes CPU consumed on other threads and cannot identify a causal library or OS layer. Executor spans include SNNLab, Python, PyTorch and compiled kernels. Start lag is measured separately, before callback entry. Deadline categories are arithmetic conditions at the measured start time, not causal attributions.",
        instrumentation="Optional --profile-timing; measurements include clock-read and storage overhead. Empty-span probes are reported, not subtracted. Profiling can change timing and spike arrival order. callback_other is the full span minus measured inner spans, including profiling overhead and diagnostic-buffer writes.",
    )


def stop(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_ros(args):
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) < 4:
        raise RuntimeError(
            "Need four allowed CPUs: one each for flexor, extensor, plant and recorder"
        )
    args.cpu_allocation = dict(zip((*NODES, "recorder"), available_cpus[:4]))
    # Set affinity before ROS starts threads. taskset assigns each child its own CPU.
    os.sched_setaffinity(0, {args.cpu_allocation["recorder"]})
    import rclpy
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Int64MultiArray, String

    class Recorder(Node):
        def __init__(self):
            super().__init__("recorder")
            self.records = {name: [] for name in NODES}
            self.spikes = {topic: [] for topic in TOPICS}
            self.handles = [
                self.create_subscription(
                    String,
                    "/diagnostics/" + name,
                    lambda msg, name=name: self.records[name].extend(json.loads(msg.data)),
                    100,
                )
                for name in NODES
            ]
            self.handles.extend(
                self.create_subscription(
                    Int64MultiArray,
                    topic,
                    lambda msg, topic=topic: self.spikes[topic].append(list(msg.data)),
                    1000,
                )
                for topic in TOPICS
            )

    output = args.output_dir / "ros2"
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    recorder = Recorder()
    children, logs = [], []
    error, ready, start_ns = None, False, 0
    try:
        for name in NODES:
            log = (output / f"node_{name}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        "taskset",
                        "--cpu-list",
                        str(args.cpu_allocation[name]),
                        sys.executable,
                        str(Path(__file__).with_name("node_" + name + ".py")),
                        "--ros-args",
                        "-p",
                        f"dt_ms:={float(args.dt_ms)}",
                        "-p",
                        f"timer_ms:={float(args.timer_ms)}",
                        "-p",
                        f"duration:={float(args.duration)}",
                        "-p",
                        f"neurons:={args.neurons}",
                        "-p",
                        f"max_lag_ms:={float(args.max_lag_ms)}",
                        "-p",
                        f"profile_timing:={str(args.profile_timing).lower()}",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        clients = {
            name: recorder.create_client(SetParameters, f"/{name}/set_parameters") for name in NODES
        }
        deadline = time.monotonic() + 90
        while not ready:
            rclpy.spin_once(recorder, timeout_sec=0.01)
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A node exited during startup; inspect node logs")
            ready = (
                set(NODES) <= set(recorder.get_node_names())
                and all(client.service_is_ready() for client in clients.values())
                and all(
                    recorder.count_publishers(topic) == 1
                    and recorder.count_subscribers(topic) >= (3 if "/motor/" in topic else 2)
                    for topic in TOPICS
                )
            )
            if time.monotonic() > deadline:
                raise TimeoutError("Nodes or spike connections did not become ready")
        # One future wall-clock origin, after discovery and SNNLab warm-up.
        start_ns = recorder.get_clock().now().nanoseconds + 2_000_000_000
        futures = [
            client.call_async(
                SetParameters.Request(
                    parameters=[Parameter("start_ns", value=start_ns).to_parameter_msg()]
                )
            )
            for client in clients.values()
        ]
        while not all(future.done() for future in futures):
            rclpy.spin_once(recorder, timeout_sec=0.01)
            if recorder.get_clock().now().nanoseconds >= start_ns:
                raise TimeoutError("Failed to arm every node before the shared start")
        for future in futures:
            response = future.result()
            if not all(result.successful for result in response.results):
                raise RuntimeError("A node rejected the shared start time")
        end_ns = start_ns + int((args.duration + 10) * 1e9)
        while not all(
            any(row["kind"] == "finished" for row in rows) for rows in recorder.records.values()
        ):
            rclpy.spin_once(recorder, timeout_sec=0.002)
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A node exited during the experiment; inspect node logs")
            if recorder.get_clock().now().nanoseconds > end_ns:
                raise TimeoutError("Nodes did not finish exporting buffered diagnostics")
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        stop(children)
        for log in logs:
            log.close()
        until = time.monotonic() + 0.2
        while time.monotonic() < until:
            rclpy.spin_once(recorder, timeout_sec=0.002)
        values, summary = collect(recorder.records, recorder.spikes, start_ns, args, ready, error)
        save(values, summary, output)
        recorder.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summary, indent=2))
    return summary


def collect(records, spikes, start_ns, args, ready, error):
    ticks = {name: [r for r in rows if r["kind"] == "tick"] for name, rows in records.items()}
    deliveries = {
        name: [r for r in rows if r["kind"] == "delivery"] for name, rows in records.items()
    }

    def relative(stamp):
        return (stamp - start_ns) / 1e9

    values = {}
    for name, rows in ticks.items():
        values[name + "_time_s"] = np.array([relative(r["stamp_ns"]) for r in rows])
        values[name + "_compute_ms"] = np.array([r["compute_ms"] for r in rows])
        values[name + "_callback_ms"] = np.array([r["callback_ms"] for r in rows])
        values[name + "_lateness_ms"] = np.array([r["deadline_lateness_ms"] for r in rows])
        values[name + "_model_time_s"] = np.array([r["model_time_s"] for r in rows])
        values[name + "_lag_ms"] = np.array([r["wall_lag_ms"] for r in rows])
        if name in SIDES:
            values[name + "_voltage"] = np.array([r["voltage"] for r in rows]).reshape(-1, 2)
            values[name + "_substep_voltage"] = np.array(
                [r["substep_voltage"] for r in rows]
            ).reshape(-1, args.substeps, 2)
            values[name + "_substep_spikes"] = np.array(
                [r["substep_spikes"] for r in rows]
            ).reshape(-1, args.substeps, 2)
            values[name + "_spikes"] = np.array([r["spikes"] for r in rows]).reshape(-1, 2)
        rows = deliveries[name]
        values[name + "_delivery_time_s"] = np.array([relative(r["stamp_ns"]) for r in rows])
        values[name + "_delivery_ms"] = np.array([r["age_ms"] for r in rows])
    plant = ticks["plant"]
    for key in ("angle", "velocity", "torque"):
        values[key] = np.array([r[key] for r in plant])
    values["activation"] = np.array([r["activation"] for r in plant]).reshape(-1, 2)
    for topic, rows in spikes.items():
        key = topic.strip("/").replace("/", "_")
        values[key + "_time_s"] = np.array([relative(r[1]) for r in rows])
        values[key + "_model_time_s"] = np.array([r[3] / 1e9 for r in rows])
        values[key + "_neuron"] = np.array([r[2] for r in rows], dtype=int)
    velocity = values["velocity"]
    moving = np.sign(velocity[np.abs(velocity) > 0.05])
    reversals = int(np.count_nonzero(np.diff(moving)))
    expected = round(args.duration * 1000 / args.timer_ms)
    progress = {name: rows[-1]["model_time_s"] if rows else 0.0 for name, rows in ticks.items()}
    missing = {name: sum(r["missing"] for r in rows) for name, rows in deliveries.items()}
    compute = {name: stats(values[name + "_compute_ms"]) for name in NODES}
    callback = {name: stats(values[name + "_callback_ms"]) for name in NODES}
    lateness = {name: stats(values[name + "_lateness_ms"]) for name in NODES}
    delivery = {name: stats(values[name + "_delivery_ms"]) for name in NODES}
    callback_overruns = {
        name: int((values[name + "_callback_ms"] > args.timer_ms).sum()) for name in NODES
    }
    overruns = {name: int((values[name + "_compute_ms"] > args.timer_ms).sum()) for name in NODES}
    age_failures = {
        name: int((values[name + "_delivery_ms"] > args.max_spike_age_ms).sum()) for name in NODES
    }
    step_late = {name: int((values[name + "_lateness_ms"] > 0).sum()) for name in NODES}
    observed_affinity = {
        name: next((r["cpu_affinity"] for r in rows if "cpu_affinity" in r), [])
        for name, rows in ticks.items()
    }
    observed_affinity["recorder"] = sorted(os.sched_getaffinity(0))
    execution = {
        name: next((r["model_info"] for r in ticks[name] if "model_info" in r), {})
        for name in SIDES
    }
    stop_reasons = {
        name: next((r["stop_reason"] for r in rows if r["kind"] == "finished"), "missing")
        for name, rows in records.items()
    }
    limit_hits = max([r["limit_hits"] for r in plant], default=0)
    checks = dict(
        nodes_ready=ready,
        one_cpu_per_node=all(
            observed_affinity[n] == [cpu] for n, cpu in args.cpu_allocation.items()
        ),
        compiled_controller_blocks=all(
            info.get("execution_path") == "compiled_blocks" for info in execution.values()
        )
        if args.substeps > 1
        else True,
        no_process_error=error is None,
        diagnostics_exported=all(reason != "missing" for reason in stop_reasons.values()),
        no_safety_stop=all(reason == "completed" for reason in stop_reasons.values()),
        all_nodes_recorded=all(ticks.values()),
        both_motor_outputs=all(spikes["/motor/" + s] for s in SIDES),
        both_sensory_streams=all(spikes["/sensory/" + s] for s in SIDES),
        repeated_movement=reversals >= 4,
        no_mechanical_stops=limit_hits == 0,
        no_detected_spike_loss=not any(missing.values()),
    )
    timing_checks = dict(
        full_model_progress=all(len(rows) >= expected - 1 for rows in ticks.values()),
        callback_within_budget=not any(callback_overruns.values()),
        compute_within_budget=not any(overruns.values()),
        steps_completed_by_deadline=not any(step_late.values()),
        spike_delivery_within_budget=not any(age_failures.values()),
    )
    profiling = (
        profile_summary(values, ticks, args.timer_ms) if args.profile_timing else {"enabled": False}
    )
    return values, dict(
        mode="ros2",
        timing_profile=profiling,
        passed=all(checks.values()),
        checks=checks,
        error=error,
        timing_passed=all(timing_checks.values()),
        timing_checks=timing_checks,
        duration_seconds=args.duration,
        dt_ms=args.dt_ms,
        timer_ms=args.timer_ms,
        substeps_per_update=args.substeps,
        neurons_per_population=args.neurons,
        total_neurons=4 * args.neurons,
        snnlab_ref=SNNLAB_REF,
        cpu_allocation=args.cpu_allocation,
        observed_cpu_affinity=observed_affinity,
        controller_execution=execution,
        model_seconds_per_node=progress,
        simulated_seconds_per_wall_second=min(progress.values()) / args.duration,
        joint_direction_reversals=reversals,
        mechanical_stop_hits=limit_hits,
        angle_range_rad=[float(values["angle"].min()), float(values["angle"].max())]
        if len(plant)
        else None,
        motor_spikes={s: len(spikes["/motor/" + s]) for s in SIDES},
        sensory_spikes={s: len(spikes["/sensory/" + s]) for s in SIDES},
        update_compute_ms=compute,
        full_callback_ms=callback,
        deadline_lateness_ms=lateness,
        callback_overruns=callback_overruns,
        schedule="Absolute releases from shared epoch; catch up without dropping windows or resetting deadlines; stop if start lag exceeds max_lag_ms",
        max_lag_ms=args.max_lag_ms,
        stop_reasons=stop_reasons,
        spike_delivery_ms=delivery,
        compute_overruns=overruns,
        step_deadline_misses=step_late,
        late_spikes=age_failures,
        missing_spikes=missing,
        spike_age_budget_ms=args.max_spike_age_ms,
        timing_scope="Live wall-clock run. Full callbacks include model work, spike publication and writes to preallocated diagnostic buffers. JSON conversion and diagnostic publication occur after the shared control interval. Deadline lateness uses full callback duration, not compute alone. ROS input callbacks execute while waiting between absolute releases; their contention appears in start lag and delivery age. Inputs enter at the first substep and outputs publish after the window. Delivery age excludes pre-publication buffering. Late spikes are still applied.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt-ms", type=float, choices=DT_CHOICES, default=0.1)
    parser.add_argument(
        "--timer-ms",
        type=float,
        default=1.0,
        help="ROS callback period, independent of numerical dt",
    )
    parser.add_argument("--neurons", type=int, default=1)
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Record per-stage wall and thread CPU times; adds measurement overhead",
    )
    parser.add_argument("--max-spike-age-ms", type=float, default=5.0)
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=20.0,
        help="Stop a model if its start lag exceeds this budget; never reset deadlines",
    )
    parser.add_argument("--basic-only", action="store_true")
    parser.add_argument("--ros2-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()
    try:
        args.substeps = window_steps(args.dt_ms, args.timer_ms)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.duration) or args.duration < 2 or args.neurons < 1:
        parser.error("Use duration >= 2 seconds and neurons >= 1")
    if not math.isfinite(args.max_spike_age_ms) or args.max_spike_age_ms <= 0:
        parser.error("Spike age budget must be positive and finite")
    if not math.isfinite(args.max_lag_ms) or args.max_lag_ms <= 0:
        parser.error("Maximum start lag must be positive and finite")
    if args.basic_only and args.ros2_only:
        parser.error("Choose only one of --basic-only and --ros2-only")
    if not args.ros2_only:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("basic_sim.py")),
                "--duration",
                str(args.duration),
                "--dt-ms",
                str(args.dt_ms),
                "--neurons",
                str(args.neurons),
                "--output-dir",
                str(args.output_dir / "basic"),
            ],
            check=True,
        )
    if not args.basic_only:
        summary = run_ros(args)
        # Timing failure is retained in the report; exit 1 also marks missed budgets.
        raise SystemExit(0 if summary["passed"] and summary["timing_passed"] else 1)


if __name__ == "__main__":
    main()
