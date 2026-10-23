FROM ros:humble-ros-base-jammy

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /uvx /usr/local/bin/

ARG SNNLAB_REF=4cffb5b78185633bb4e9b898430b04bfc6c49908
ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    SNNLAB_PYTHON=/opt/snnlab-venv/bin/python \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-colcon-common-extensions \
    python3-matplotlib python3-numpy ros-humble-rclpy ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

# Keep SNNLab's NumPy 2 separate from Humble's system Python extensions.
RUN uv venv --python /usr/bin/python3 /opt/snnlab-venv \
    && uv pip install --python /opt/snnlab-venv/bin/python \
       torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /opt/snnlab-venv/bin/python \
       "snnlab @ git+https://github.com/eoinmurray/snnlab.git@${SNNLAB_REF}"
ENV SNNLAB_REF=${SNNLAB_REF}

WORKDIR /opt/ros_ws
COPY src/ src/
RUN . /opt/ros/humble/setup.sh && colcon build --executor sequential
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "snnlab_ros2", "experiment", "--output-dir", "/artifacts"]
