"""Network definition and stateful execution, independent of ROS and IPC."""

import math
import time

import torch
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph


DT_MS = 1.0
STEPS = 10
TONIC_TAU_MS = 1.0
EVENT_TAU_MS = 5.0
EVENT_WEIGHT_NA = 2.0


def build_network():
    """Build the demonstration LIF network using SNNLab's authoring API."""
    net = lang.Network("ros_lif", dt=DT_MS * lang.ms)
    tonic = net.input(
        "tonic", shape=("time", "batch", 1), signal_type="current", unit="nA"
    )
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    cell = net.population(
        "cell",
        size=1,
        neuron=lang.CUBA_LIF(
            tau_mem=20 * lang.ms, capacitance_nf=0.1, refractory=2 * lang.ms
        ),
    )
    # This is filtered tonic drive, rather than instantaneous current injection.
    # Normalize the recurrence so its steady-state output equals the input nA.
    net.connect(
        tonic,
        cell.excitatory,
        name="tonic_drive",
        synapse=lang.ExponentialCurrent(tau=TONIC_TAU_MS * lang.ms),
        weight=lang.Constant(1 - math.exp(-DT_MS / TONIC_TAU_MS)),
        initialization_scaling="direct",
    )
    inhibition = net.connect(
        events,
        cell.inhibitory,
        name="event_drive",
        synapse=lang.ExponentialCurrent(tau=EVENT_TAU_MS * lang.ms),
        weight=lang.Constant(EVENT_WEIGHT_NA),
        initialization_scaling="direct",
    )
    net.output("voltage", cell.voltage)
    net.output("spikes", cell.spikes)
    net.output("inhibitory_current", inhibition.current)
    return net


class Simulation:
    """Advance fixed windows while carrying SNNLab's complete runtime state."""

    def __init__(self):
        bundle = lang.compile(build_network())
        self.executor = GraphExecutor(
            plan_graph(bundle.graph), compile_mode=None, cuda_graphs="off"
        ).eval()
        self.state = None
        # Warm the executor, then discard the warmup's simulation state.
        self.step(0.0, 0)
        self.state = None

    @torch.inference_mode()
    def step(self, current_na, event_count):
        """Apply held current and a batch of coincident predecessor events.

        The single event channel contains a count at the first timestep, which
        sums simultaneous impulses into the linear inhibitory synapse. Counts
        can exceed one; this is not a binary per-neuron spike raster and does
        not preserve arrival times within a window.
        """
        tonic = torch.full((STEPS, 1, 1), current_na, dtype=torch.float32)
        events = torch.zeros_like(tonic)
        events[0, 0, 0] = event_count
        started = time.perf_counter_ns()
        result = self.executor(
            {"tonic": tonic, "events": events},
            runtime_state=self.state,
            diagnostics=False,
        )
        self.state = result.runtime_state
        return {
            "voltage_mv": float(result.outputs["voltage"][-1, 0, 0]),
            "spike_offsets": torch.nonzero(
                result.outputs["spikes"][:, 0, 0], as_tuple=False
            ).flatten().tolist(),
            "inhibitory_peak_na": float(result.outputs["inhibitory_current"].max()),
            "completed_steps": int(self.state.completed_steps),
            "compute_ms": (time.perf_counter_ns() - started) / 1e6,
        }
