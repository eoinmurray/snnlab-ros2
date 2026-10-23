"""All plots for the standalone and distributed runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recipe import Joint

COLOURS = ("#2865a0", "#c06b26")
SIDES = ("flexor", "extensor")


def label_panels(axes):
    """Keep panel letters outside the axes, with consistent physical spacing."""
    for index, axis in enumerate(axes):
        axis.annotate(
            chr(ord("A") + index),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-10, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            annotation_clip=False,
        )


def plot_basic(values, summary, output):
    t = values["time_s"]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 12),
        sharex=True,
        height_ratios=[0.8, 1.4, 0.8, 1.2, 1.1],
        constrained_layout=True,
    )
    # Sensory events are supplied at the start of a step; state is sampled at its end.
    input_t = t - summary["dt_ms"] / 1000
    for i, (side, colour) in enumerate(zip(SIDES, COLOURS)):
        axes[0].scatter(
            input_t[values["sensory"][:, i] > 0],
            np.full(int(values["sensory"][:, i].sum()), i),
            marker="|",
            s=45,
            color=colour,
        )
        for cell, linestyle in ((0, "-"), (1, "--")):
            column = i * 2 + cell
            axes[1].plot(
                t,
                values["voltage"][:, column],
                color=colour,
                lw=0.7,
                ls=linestyle,
                alpha=0.85,
                label=f"{side} {'motor' if cell == 0 else 'slow'}",
            )
        motor = values["spikes"][:, i * 2] > 0
        axes[2].scatter(t[motor], np.full(int(motor.sum()), i), marker="|", s=45, color=colour)
    for row in (0, 2):
        axes[row].set_yticks([0, 1], ["flexor", "extensor"])
        axes[row].set_ylim(-0.5, 1.5)
    axes[0].set_ylabel("Sensory spikes\nto controllers")
    axes[1].set_ylabel("Population mean\nvoltage (mV)")
    axes[1].legend(loc="lower right", ncol=4, fontsize=8)
    axes[2].set_ylabel("Motor output\nspikes")
    axes[3].plot(t, values["angle"], color="#232e40", lw=1.3, label="joint angle")
    axes[3].axhline(Joint.sensory_limit, color="grey", ls=":", lw=0.9, label="sensory limits")
    axes[3].axhline(-Joint.sensory_limit, color="grey", ls=":", lw=0.9)
    axes[3].set_ylim(-1.1, 1.1)
    axes[3].set_ylabel("Joint angle\n(rad)")
    muscle = axes[3].twinx()
    for i, (side, colour) in enumerate(zip(SIDES, COLOURS)):
        muscle.plot(
            t,
            values["activation"][:, i],
            color=colour,
            alpha=0.6,
            lw=0.8,
            label=f"{side} activation",
        )
    muscle.set_ylim(0, 1)
    muscle.set_ylabel("Muscle activation\n(dimensionless)")
    lines = axes[3].get_lines() + muscle.get_lines()
    lines = [line for line in lines if not line.get_label().startswith("_")]
    axes[3].legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower right",
        ncol=2,
        fontsize=8,
    )
    for field, colour, label in [
        ("controller_ms", COLOURS[0], "SNNLab update"),
        ("plant_ms", COLOURS[1], "plant update"),
        ("loop_ms", "#232e40", "whole step"),
    ]:
        axes[4].plot(t, values[field], lw=0.6, alpha=0.75, color=colour, label=label)
    axes[4].axhline(summary["dt_ms"], color="#aa3333", ls="--", lw=1, label="simulated step budget")
    axes[4].set_ylabel("Compute time\n(ms)")
    axes[4].legend(loc="upper right", ncol=4, fontsize=8)
    axes[4].set_xlabel("Simulation time (s) — unpaced execution, not wall time")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.set_xlim(0, summary["simulated_seconds"])
    fig.suptitle(
        f"{summary.get('mode', 'basic').upper()}: reciprocal controller → mechanical joint\n"
        f"{summary['motor_dominance_handoffs']} motor handoffs · "
        f"{summary['joint_direction_reversals']} movement reversals · "
        f"{summary.get('simulated_seconds_per_wall_second', 1):.2f}× model / wall time",
        fontsize=14,
    )
    label_panels(axes)
    fig.savefig(output / "experiment.png", dpi=150)
    plt.close(fig)


def plot_ros_timing(fig, slots, shared_axis, values, summary):
    """Expose schedule offset, absolute deadlines and compute/transport separately."""
    lag_axis = fig.add_subplot(slots[0], sharex=shared_axis)
    late_axis = fig.add_subplot(slots[1], sharex=shared_axis)
    table_axis = fig.add_subplot(slots[2])
    timing = "PASS" if summary["timing_passed"] else "FAIL"
    checks = summary["timing_checks"]
    status = " · ".join(
        f"{label} {'PASS' if checks[key] else 'FAIL'}"
        for label, key in (
            ("Progress", "full_model_progress"),
            ("Compute", "compute_within_budget"),
            ("Callbacks", "callback_within_budget"),
            ("Deadlines", "steps_completed_by_deadline"),
            ("Delivery", "spike_delivery_within_budget"),
        )
        if key in checks
    )
    lag_axis.set_title(f"Timing {timing} — {status}", loc="left", fontsize=10, pad=10)
    rows = []
    for name, colour in zip((*SIDES, "plant"), (*COLOURS, "#267263")):
        time_s = values[name + "_time_s"]
        lag = values[name + "_lag_ms"]
        compute = values[name + "_compute_ms"]
        # Same deadline convention as test.py; do not reset the schedule after lateness.
        lateness = values.get(name + "_lateness_ms", lag + compute - summary["timer_ms"])
        lag_axis.plot(time_s, lag, lw=0.7, color=colour, label=name)
        late_axis.plot(time_s, lateness, lw=0.7, color=colour)
        n = len(compute)
        misses = summary["step_deadline_misses"][name]
        stats = summary["update_compute_ms"][name]
        callback = summary.get("full_callback_ms", {}).get(name)
        delivery_stats = summary["spike_delivery_ms"][name]
        delivery = delivery_stats["maximum"] if delivery_stats else None
        rows.append(
            [
                name.capitalize(),
                f"{stats['median']:.3f} / {stats['p95']:.3f} / {stats['maximum']:.3f}"
                if stats
                else "—",
                f"{callback['p95']:.3f} / {callback['maximum']:.3f}" if callback else "—",
                f"{summary['compute_overruns'][name]:,} / {n:,}",
                f"{summary['callback_overruns'][name]:,} / {n:,}" if callback else "—",
                f"{delivery:.3f}" if delivery is not None else "—",
                f"{misses:,} / {n:,} ({misses / n:.2%})" if n else "—",
            ]
        )
    lag_axis.set_ylabel("Start lag\n(ms)")
    lag_axis.axhline(0, color="#555555", ls="--", lw=0.8)
    lag_axis.legend(loc="upper right", ncol=3, fontsize=8)
    lag_axis.tick_params(labelbottom=False)
    late_axis.set_ylabel("Deadline lateness\n(ms)")
    late_axis.axhline(0, color="#aa3333", lw=1)
    low, high = late_axis.get_ylim()
    late_axis.axhspan(0, max(high, 0.1), color="#c83333", alpha=0.09, zorder=0)
    late_axis.set_ylim(min(low, -0.1), max(high, 0.1))
    late_axis.text(
        0.995,
        0.96,
        "Above zero = missed deadline",
        transform=late_axis.transAxes,
        ha="right",
        va="top",
        color="#9b2727",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    late_axis.set_xlabel("Wall time since the shared start (s)")
    table_axis.set_axis_off()
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Node",
            "Compute median / p95 / max (ms)",
            "Full callback p95 / max (ms)",
            f"Compute > {summary['timer_ms']:g} ms",
            f"Callback > {summary['timer_ms']:g} ms",
            "Delivery max (ms)",
            "Deadlines missed",
        ],
        colWidths=[0.07, 0.22, 0.17, 0.10, 0.10, 0.11, 0.23],
        cellLoc="center",
        bbox=[0, 0.26, 1, 0.67],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_text_props(weight="bold")
        else:
            name = (*SIDES, "plant")[row - 1]
            failed = (
                (column == 3 and summary["compute_overruns"][name] > 0)
                or (column == 4 and summary.get("callback_overruns", {}).get(name, 0) > 0)
                or (column == 5 and summary["late_spikes"][name] > 0)
                or (column == 6 and summary["step_deadline_misses"][name] > 0)
            )
            if failed:
                cell.set_facecolor("#fbe6e6")
                cell.set_text_props(color="#9b2727", weight="bold")
    table_axis.text(
        0,
        0.17,
        "Start lag: callback start minus scheduled window start.  "
        + (
            "Lateness: full callback end minus scheduled window end."
            if "full_callback_ms" in summary
            else "Lateness: measured compute end minus scheduled window end."
        ),
        transform=table_axis.transAxes,
        fontsize=8,
        va="top",
    )
    table_axis.text(
        0,
        0.03,
        f"Red cells fail a budget. Compute ≤ {summary['timer_ms']:g} ms; "
        f"delivery ≤ {summary['spike_age_budget_ms']:g} ms; deadlines ≤ 0 ms late. "
        "Full callback includes diagnostic buffering; delivery excludes pre-publication buffering.",
        transform=table_axis.transAxes,
        fontsize=8,
        va="top",
    )
    for axis in (lag_axis, late_axis):
        axis.grid(alpha=0.18)
        axis.set_xlim(0, summary["duration_seconds"])
    return lag_axis


def plot_ros(values, summary, output):
    fig = plt.figure(figsize=(14, 16), constrained_layout=True)
    grid = fig.add_gridspec(7, 1, height_ratios=[0.8, 1.4, 0.8, 1.2, 0.65, 0.65, 0.95])
    axes = [fig.add_subplot(grid[0])]
    axes.extend(fig.add_subplot(grid[row], sharex=axes[0]) for row in range(1, 4))
    for i, (side, colour) in enumerate(zip(SIDES, COLOURS)):
        inputs = values["sensory_" + side + "_time_s"]
        outputs = values["motor_" + side + "_time_s"]
        axes[0].scatter(inputs, np.full(len(inputs), i), marker="|", s=40, color=colour)
        for j, style in [(0, "-"), (1, "--")]:
            axes[1].plot(
                values[side + "_time_s"],
                values[side + "_voltage"][:, j],
                color=colour,
                ls=style,
                lw=0.7,
                label=side + (" motor" if j == 0 else " slow"),
            )
        axes[2].scatter(outputs, np.full(len(outputs), i), marker="|", s=40, color=colour)
    for row in (0, 2):
        axes[row].set_yticks([0, 1], SIDES)
        axes[row].set_ylim(-0.5, 1.5)
    axes[0].set_ylabel("Sensory spikes\nto controllers")
    axes[1].set_ylabel("Population mean\nvoltage (mV)")
    axes[1].legend(loc="lower right", ncol=4, fontsize=8)
    axes[2].set_ylabel("Motor output\nspikes")
    # Give angle and activation distinct axes, as in the standalone figure.
    axes[3].plot(values["plant_time_s"], values["angle"], color="#232e40", label="joint angle")
    for sign in (-1, 1):
        axes[3].axhline(sign * Joint.sensory_limit, color="grey", ls=":", lw=0.9)
    axes[3].set_ylabel("Joint angle\n(rad)")
    axes[3].set_ylim(-1.1, 1.1)
    muscle = axes[3].twinx()
    for i, (side, colour) in enumerate(zip(SIDES, COLOURS)):
        muscle.plot(
            values["plant_time_s"],
            values["activation"][:, i],
            color=colour,
            lw=0.8,
            alpha=0.6,
            label=side + " activation",
        )
    muscle.set_ylim(0, 1)
    muscle.set_ylabel("Muscle activation\n(dimensionless)")
    lines = axes[3].get_lines()[:1] + muscle.get_lines()
    axes[3].legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower right",
        ncol=3,
        fontsize=8,
    )
    timing_axis = plot_ros_timing(fig, [grid[row] for row in range(4, 7)], axes[0], values, summary)
    for axis in axes:
        axis.tick_params(labelbottom=False)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.set_xlim(0, summary["duration_seconds"])
    behaviour = "PASS" if summary["passed"] else "FAIL"
    timing = "PASS" if summary["timing_passed"] else "FAIL"
    fig.suptitle(
        f"Three ROS processes · {summary['total_neurons']} neurons · dt {summary['dt_ms']} ms / window {summary['timer_ms']} ms · behaviour {behaviour} · timing {timing}\n"
        f"{summary['joint_direction_reversals']} movement reversals · "
        "model seconds: "
        + ", ".join(
            f"{name} {seconds:.2f}" for name, seconds in summary["model_seconds_per_node"].items()
        ),
        fontsize=13,
    )
    label_panels([*axes, timing_axis])
    fig.savefig(output / "experiment.png", dpi=150)
    plt.close(fig)


def plot_timing_profile(summary, output):
    """Show time inside each boundary without assigning blame for elapsed stalls."""
    profile = summary["timing_profile"]
    nodes = list(profile["nodes"])
    fig, axes = plt.subplots(
        len(nodes), 2, figsize=(14, 3.1 * len(nodes) + 1), squeeze=False, constrained_layout=True
    )
    labels = {
        "callback": "Full callback",
        "snnlab_executor": "SNNLab executor call",
        "input_preparation": "Input preparation",
        "output_extraction": "Output extraction",
        "plant_dynamics": "Plant dynamics",
        "spike_publication": "Spike publication",
        "callback_other": "Other / measurement",
    }
    for row, name in enumerate(nodes):
        data = profile["nodes"][name]
        stages = ["callback", *data["stage_order"][:-1], "callback_other"]
        maxima = []
        for column, subset in enumerate(("all_updates", "missed_deadlines")):
            axis = axes[row, column]
            samples = data["stages"]["callback"][subset]
            count = samples["wall_ms"]["n"] if samples else 0
            title = "all updates" if column == 0 else "missed deadlines only"
            axis.set_title(f"{name.capitalize()} — {title} (n={count:,})", loc="left", fontsize=11)
            if not count:
                axis.text(0.5, 0.5, "No samples", transform=axis.transAxes, ha="center")
                continue
            wall = np.array(
                [data["stages"][stage][subset]["wall_ms"]["mean"] * 1000 for stage in stages]
            )
            cpu = np.array(
                [data["stages"][stage][subset]["thread_cpu_ms"]["mean"] * 1000 for stage in stages]
            )
            y = np.arange(len(stages))
            axis.barh(y - 0.18, wall, height=0.34, color="#b7cfe3", label="Elapsed wall time")
            axis.barh(y + 0.18, cpu, height=0.34, color="#2865a0", label="Calling-thread CPU time")
            for index, (w, c) in enumerate(zip(wall, cpu)):
                axis.annotate(
                    f"{w:.1f} / {c:.1f}",
                    (max(w, c), index),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            axis.set_yticks(y, [labels[stage] for stage in stages])
            axis.invert_yaxis()
            axis.set_xlabel("Mean duration (µs); labels are wall / thread CPU")
            axis.grid(axis="x", alpha=0.18)
            maxima.append(max(wall.max(), cpu.max()))
        limit = max(maxima, default=1) * 1.24
        for axis in axes[row]:
            axis.set_xlim(0, limit)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Where callback time goes — optional instrumentation enabled\n"
        "The executor boundary includes SNNLab, Python, PyTorch and compiled kernels; elapsed stalls are not library blame.",
        fontsize=12,
    )
    fig.supxlabel(
        "Full callback is a total, not another stage. Wall − thread CPU suggests blocking/descheduling plus probe error; other threads are excluded.\n"
        "Measurements include profiler overhead. Averages in the right column describe the missed windows in this run, not a causal experiment.",
        fontsize=9,
    )
    fig.savefig(output / "timing_profile.png", dpi=150)
    plt.close(fig)


def plot(values, summary, output):
    (plot_ros if summary.get("mode") == "ros2" else plot_basic)(values, summary, output)
    if summary.get("timing_profile", {}).get("enabled"):
        plot_timing_profile(summary, output)


def save(values, summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "traces.npz", **values)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(values, summary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot(
        dict(np.load(args.directory / "traces.npz")),
        json.loads((args.directory / "summary.json").read_text()),
        args.directory,
    )


if __name__ == "__main__":
    main()
