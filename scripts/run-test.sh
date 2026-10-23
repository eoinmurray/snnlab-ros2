#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SNNLAB_ROS2_IMAGE:-snnlab-ros2:humble}"
output_root="$repo_root/artifacts"
mkdir -p "$output_root/runs"

if ! docker info >/dev/null 2>&1; then
    echo 'Docker is unavailable. Start your Docker engine first.' >&2
    exit 1
fi

# Docker's layer cache rebuilds only changed layers. No Compose dependency.
docker build --tag "$image" "$repo_root"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$output_root/runs/$run_id"
mkdir -p "$run_dir"
container_id="$(docker create --init --network none \
    --env ROS_LOCALHOST_ONLY=1 --env ROS_DOMAIN_ID=71 \
    "$image" ros2 run snnlab_ros2 experiment --output-dir /artifacts "$@")"
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT

status=0
docker start --attach "$container_id" || status=$?
container_status="$(docker inspect --format '{{.State.ExitCode}}' "$container_id")"
if [[ "$container_status" != 0 ]]; then
    status="$container_status"
fi
# Copy from the stopped container, including partial failure diagnostics.
if docker cp "$container_id:/artifacts/." "$run_dir/"; then
    if [[ -f "$run_dir/experiment.png" ]]; then
        cp "$run_dir/experiment.png" "$output_root/experiment.png"
    fi
    if [[ -f "$run_dir/summary.json" ]]; then
        cp "$run_dir/summary.json" "$output_root/summary.json"
    fi
    if [[ -f "$run_dir/summary.json" && -f "$run_dir/experiment.png" ]]; then
        python3 - "$repo_root" <<'PY'
import json
from datetime import datetime
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / 'artifacts/summary.json').read_text())
status = 'passed' if summary['passed'] else 'failed'
date = datetime.fromisoformat(summary['created_utc']).strftime('%d %B %Y')
checks = summary['checks']
metrics = summary['metrics']
parts = [
    f"The latest test **{status}** on {date}. "
    f"{sum(checks.values())} of {len(checks)} checks passed. "
    f"{summary['input_events_acked']} of {summary['input_events_sent']} input events were processed.",
    '![Measured ROS 2 experiment](artifacts/experiment.png)',
]
if summary['passed']:
    spikes = metrics['spikes_by_phase']
    parts.append(
        f"The neuron fired {spikes.get('driven', 0)} times with current applied, "
        f"and {spikes.get('inhibited', 0)} times when inhibitory inputs were added. "
        f"It fired {spikes.get('baseline', 0)} times before the current was applied "
        f"and {spikes.get('recovery', 0)} times after it was removed."
    )
timing = metrics.get('adapter_ms')
if timing:
    parts.append(
        f"Each update advances {summary['steps_per_tick'] * summary['dt_ms']:g} ms "
        f"of simulated time. 95% of updates took at most {timing['p95']:.2f} ms, "
        f"including communication with the simulation worker. "
        f"{metrics['adapter_overruns']} of {timing['n']} updates took longer than "
        f"the {summary['wall_timer_ms']:g} ms target; the slowest took {timing['max']:.2f} ms. "
        "This does not guarantee that every future update will finish on time."
    )
if not summary['passed']:
    parts.append('See the summary and per-run logs for the failure details.')
parts.append('[Full results](artifacts/summary.json)')
block = '\n\n'.join(parts) + '\n'
readme = root / 'README.md'
text = readme.read_text()
start = '<!-- experiment-result:start -->'
end = '<!-- experiment-result:end -->'
before, remainder = text.split(start, 1)
_, after = remainder.split(end, 1)
readme.write_text(before + start + '\n' + block + end + after)
PY
    fi
    echo "Artifacts: $run_dir"
else
    echo 'No artifacts could be extracted; inspect the container output above.' >&2
    status=1
fi
exit "$status"
