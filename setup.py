from setuptools import setup

setup(
    name='cycle-gan',
    version="0.0.1",
    packages=["cycle_gan"],
    install_requires=["numpy", "matplotlib", "tensorflow~=2.3"],
)
