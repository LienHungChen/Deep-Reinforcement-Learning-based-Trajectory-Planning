import os
from setuptools import setup, find_packages

with open(os.path.join("requirements", "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="DRL_b_TP",
    author="Lien-Hung Chen",
    description="Deep Learning based Trajectory Planning",
    packages=find_packages(),
    install_requires=requirements
)