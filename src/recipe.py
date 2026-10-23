"""Shared models: one two-population SNNLab controller and a mechanical joint.

Neural units are ms, mV, uS and nF. Mechanical units are seconds, radians,
kg m² and N m. ROS imports belong in the node files, not in this recipe.
"""

import math
import time

import numpy as np

SIDES = ("flexor", "extensor")
DT_CHOICES = (0.1, 0.25, 0.5, 1.0, 2.0)
SNNLAB_REF = "d2c7e82c1460fb3ee6e0f473331f0c3fe4db05fb"


def start_measurement():
    """Pair elapsed wall time with CPU time consumed by this calling thread."""
    return time.perf_counter_ns(), time.thread_time_ns()


def finish_measurement(start, target):
    cpu = time.thread_time_ns()
    wall = time.perf_counter_ns()
    target[0] = (wall - start[0]) / 1e6
    target[1] = (cpu - start[1]) / 1e6


def window_steps(dt_ms, timer_ms):
    """Require a whole number of numerical steps in each callback window."""
    if dt_ms not in DT_CHOICES or not math.isfinite(timer_ms) or timer_ms <= 0:
        raise ValueError("Choose a supported dt and a positive, finite timer period")
    ratio = timer_ms / dt_ms
    if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
        raise ValueError("Timer period must be an integer multiple of the simulation step")
    return round(ratio)


class Controller:
    """Motor and slow inhibitory populations, preserving SNNLab state each step.

    Population size one gives two neurons per controller. Larger populations
    replicate identical cells with normalized dense connectivity: a load test,
    not a claim of additional biological complexity. Only spike events connect
    this model to its peer and the plant. Tonic excitation is local.
    """

    def __init__(
        self,
        side,
        dt_ms=0.1,
        neurons=1,
        *,
        warm_steps=1,
        compile_mode="default",
        profile_timing=False,
    ):
        import torch
        from snnlab import lang
        from snnlab.sim import Performance, prepare_stream

        if side not in SIDES or dt_ms not in DT_CHOICES or neurons < 1 or warm_steps < 1:
            raise ValueError("Invalid side, timestep or population size")
        torch.set_num_threads(1)
        self.side, self.dt_ms, self.neurons = side, dt_ms, neurons
        self.profile_timing = profile_timing
        self.stage_timing = np.zeros((3, 2))
        self.torch, self.steps = torch, 0
        net = lang.Network(side + "_controller", dt=dt_ms * lang.ms)

        def connect(source, target, name, weight, tau, *, inhibitory=False, recurrent=False):
            return net.connect(
                source,
                target,
                name=name,
                synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
                weight=lang.Constant(weight),
                initialization_scaling="direct",
                constraint=lang.NonNegative(),
                **({"connection": "recurrent", "delay": dt_ms * lang.ms} if recurrent else {}),
            )

        motor = net.population(
            "motor",
            size=neurons,
            neuron=lang.COBA_LIF(tau_mem=20 * lang.ms, capacitance_nf=1, refractory=2 * lang.ms),
        )
        slow = net.population(
            "slow",
            size=neurons,
            neuron=lang.COBA_LIF(tau_mem=80 * lang.ms, capacitance_nf=1, refractory=10 * lang.ms),
        )
        tonic = net.input("tonic", shape=("time", "batch", 1), signal_type="continuous", unit="uS")
        sensory = net.input(
            "sensory", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
        )
        peer = net.input("peer", shape=("time", "batch", 1), signal_type="spikes", unit="spike")
        connect(tonic, motor.excitatory, "local_bias", 1, 0.001)
        connect(sensory, motor.excitatory, "sensory_drive", 0.03, 10)
        connect(
            peer,
            motor.inhibitory,
            "peer_inhibition",
            0.1 / neurons,
            20,
            inhibitory=True,
        )
        connect(
            motor.spikes,
            slow.excitatory,
            "recruit_slow",
            0.002 / neurons,
            80,
            recurrent=True,
        )
        connect(
            slow.spikes,
            motor.inhibitory,
            "adaptation",
            0.04 / neurons,
            250,
            inhibitory=True,
            recurrent=True,
        )
        for name, cell in [("motor", motor), ("slow", slow)]:
            net.output(name + "_voltage", cell.voltage)
            net.output(name + "_spikes", cell.spikes)
        # Preparation privately owns the model/state and caches fixed graph metadata.
        # Match the existing float64 CPU dynamics and compiled callback block size.
        started = time.perf_counter()
        self.stream = prepare_stream(
            net,
            window_steps=warm_steps,
            batch_size=1,
            device="cpu",
            dtype=torch.float64,
            performance=Performance(
                compile_mode=compile_mode,
                compile_block_steps=min(warm_steps, 64),
                cuda_graphs="off",
                precision="highest",
                projection_mode="fast",
            ),
        )
        self.inputs = {
            name: torch.zeros(warm_steps, 1, 1, dtype=torch.float64)
            for name in ("tonic", "sensory", "peer")
        }
        path = self.stream.info["execution_path"]
        if compile_mode is not None and warm_steps > 1 and path["selected"] != "compiled_blocks":
            raise RuntimeError(f"Compiled blocks unavailable: {path}")
        self.info = dict(
            prepared_stream=True,
            compile_mode=compile_mode,
            execution_path=path["selected"],
            block_steps=path["block_steps"],
            warm_input_steps=warm_steps,
            warmup_seconds=time.perf_counter() - started,
            torch_version=torch.__version__,
        )
        # Warm input preparation and output extraction too, then reset all model state.
        self.stream.reset()
        self.steps = 0
        self.window(warm_steps, sensory=1, peer=1)
        self.window(warm_steps, sensory=1, peer=1)
        self.info["warmup_seconds"] = time.perf_counter() - started
        self.stream.reset()
        self.steps = 0

    def advance(self, sensory, peer):
        """Execute a window in one SNNLab call; inputs are counts per substep."""
        length = len(sensory)
        if length < 1 or len(peer) != length:
            raise ValueError("Input windows must have the same positive length")
        with self.torch.inference_mode():
            if self.inputs["tonic"].shape[0] != length:
                raise ValueError("Input window must match the prepared stream shape")
            self.inputs["tonic"].fill_(0.03)
            if self.side == "flexor":
                bias_steps = max(0, min(length, round(20 / self.dt_ms) - self.steps))
                self.inputs["tonic"][:bias_steps] += 0.02
            for name, values in [("sensory", sensory), ("peer", peer)]:
                self.inputs[name].copy_(
                    self.torch.as_tensor(values, dtype=self.torch.float64).reshape(length, 1, 1)
                )
            return self._execute(length)

    def _execute(self, length):
        with self.torch.inference_mode():
            measured = start_measurement() if self.profile_timing else None
            result = self.stream.advance(self.inputs)
            if measured is not None:
                finish_measurement(measured, self.stage_timing[1])
            measured = start_measurement() if self.profile_timing else None
            self.steps += length
            # Outputs are CPU tensors. Reduce these tiny diagnostic arrays in NumPy
            # rather than dispatching four extra Torch operations every window.
            voltages = [
                result.outputs[name + "_voltage"][:, 0].numpy() for name in ("motor", "slow")
            ]
            spike_arrays = [
                result.outputs[name + "_spikes"][:, 0].numpy() for name in ("motor", "slow")
            ]
            voltage = np.column_stack(
                [v[:, 0] if self.neurons == 1 else v.mean(axis=1) for v in voltages]
            )
            spikes = np.column_stack(
                [v[:, 0] if self.neurons == 1 else v.sum(axis=1) for v in spike_arrays]
            ).astype(int)
            # Each pair preserves its substep index and neuron identity.
            motor_events = np.argwhere(spike_arrays[0] != 0).tolist()
            if measured is not None:
                finish_measurement(measured, self.stage_timing[2])
        return voltage, spikes, motor_events

    def step(self, sensory=0, peer=0):
        voltage, spikes, events = self.advance([sensory], [peer])
        return voltage[0], spikes[0], [neuron for _, neuron in events]

    def window(self, steps, sensory=0, peer=0):
        # ROS uses a fixed shape: reuse tensors and avoid NumPy-to-Torch conversion.
        measured = start_measurement() if self.profile_timing else None
        if steps != self.inputs["tonic"].shape[0]:
            raise ValueError("Window must match the warmed callback shape")
        with self.torch.inference_mode():
            self.inputs["tonic"].fill_(0.03)
            if self.side == "flexor" and self.steps < round(20 / self.dt_ms):
                bias_steps = min(steps, round(20 / self.dt_ms) - self.steps)
                self.inputs["tonic"][:bias_steps].fill_(0.05)
            self.inputs["sensory"].zero_()
            self.inputs["peer"].zero_()
            self.inputs["sensory"][0, 0, 0] = sensory
            self.inputs["peer"][0, 0, 0] = peer
            if measured is not None:
                finish_measurement(measured, self.stage_timing[0])
            return self._execute(steps)


class Joint:
    """Damped rotary joint with opposing spike-driven virtual muscles."""

    inertia = 0.005
    damping = 0.02
    stiffness = 0.02
    max_torque = 0.8
    activation_tau = 0.08
    sensory_limit = 0.25
    hard_limit = 1.0
    sensory_interval = 0.01

    def __init__(self):
        self.angle, self.velocity = 0.0, 0.0
        self.activation = np.zeros(2)
        self.next_sensory = np.zeros(2)
        self.limit_hits = 0

    def encode(self, now, feedback=True):
        spikes = np.zeros(2, dtype=int)
        if feedback:
            for i, active in enumerate(
                (self.angle < -self.sensory_limit, self.angle > self.sensory_limit)
            ):
                if active and now + 1e-12 >= self.next_sensory[i]:
                    spikes[i] = 1
                    self.next_sensory[i] = now + self.sensory_interval
        return spikes

    def advance(self, motor_spikes, dt):
        # Callers normalize motor counts by population size.
        self.activation = np.minimum(
            1.0,
            self.activation * math.exp(-dt / self.activation_tau) + 0.08 * np.asarray(motor_spikes),
        )
        torque = self.max_torque * (self.activation[0] - self.activation[1])
        acceleration = (
            torque - self.damping * self.velocity - self.stiffness * self.angle
        ) / self.inertia
        self.velocity += dt * acceleration
        self.angle += dt * self.velocity
        if abs(self.angle) > self.hard_limit:
            self.limit_hits += 1
            self.angle = float(np.clip(self.angle, -self.hard_limit, self.hard_limit))
            self.velocity = 0.0
        return torque
