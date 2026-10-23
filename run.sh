#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${SNNLAB_ROS2_IMAGE:-snnlab-ros2:demo}"
docker_cli=(docker)
if [[ -n "${SNNLAB_DOCKER_CONTEXT:-}" ]]; then
    docker_cli+=(--context "$SNNLAB_DOCKER_CONTEXT")
elif docker context inspect colima-snnlab-ros2 >/dev/null 2>&1; then
    docker_cli+=(--context colima-snnlab-ros2)
fi
available_cpus="$("${docker_cli[@]}" info --format '{{.NCPU}}')"
if (( available_cpus < 4 )); then
    echo 'Allocate at least four Docker CPUs: flexor, extensor, plant and recorder each need one.' >&2
    echo 'On Colima: colima start --profile snnlab-ros2 --cpus 4 --memory 4 --disk 20 --activate=false' >&2
    exit 1
fi
"${docker_cli[@]}" build --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from "$image" --tag "$image" "$repo_root"
container_id="$("${docker_cli[@]}" create --init --network none --cpuset-cpus 0-3 --cpus 4 \
    --env ROS_LOCALHOST_ONLY=1 --env ROS_DOMAIN_ID=73 "$image" "$@")"
trap '"${docker_cli[@]}" rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
status=0
"${docker_cli[@]}" start --attach "$container_id" || status=$?
container_status="$("${docker_cli[@]}" inspect --format '{{.State.ExitCode}}' "$container_id")"
if [[ "$container_status" != 0 ]]; then status="$container_status"; fi
mkdir -p "$repo_root/artifacts"
if ! "${docker_cli[@]}" cp "$container_id:/app/artifacts/." "$repo_root/artifacts/"; then
    echo 'The container did not produce results; inspect the error above.' >&2
    exit 1
fi
echo "Results: $repo_root/artifacts/basic and $repo_root/artifacts/ros2"
exit "$status"
