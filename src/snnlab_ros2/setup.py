from setuptools import find_packages, setup

setup(
    name="snnlab_ros2",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/snnlab_ros2"]),
        ("share/snnlab_ros2", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    maintainer="Eoin Murray",
    maintainer_email="me@eoinmurray.info",
    description="SNNLab streaming adapter and ROS 2 integration experiment",
    license="MIT",
    entry_points={"console_scripts": [
        "neuron = snnlab_ros2.node:main",
        "experiment = snnlab_ros2.experiment:main",
    ]},
)
