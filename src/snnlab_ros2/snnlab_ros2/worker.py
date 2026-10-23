"""JSON-lines worker, executed by the SNNLab venv rather than ROS Python."""

import contextlib
import json
import math
import sys

import torch

from snnlab_ros2.simulation import DT_MS, EVENT_WEIGHT_NA, STEPS, Simulation


def main():
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(sys.stderr):
        simulation = Simulation()
    print(json.dumps({"ready": True, "dt_ms": DT_MS, "steps": STEPS,
                      "event_weight_na": EVENT_WEIGHT_NA}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        current = float(request["current_na"])
        count = int(request["event_count"])
        if not math.isfinite(current) or current < 0 or not 0 <= count <= 1000:
            raise ValueError("Invalid current or event count")
        with contextlib.redirect_stdout(sys.stderr):
            result = simulation.step(current, count)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
