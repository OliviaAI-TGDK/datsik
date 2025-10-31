from setuptools import setup, find_packages

setup(
    name="datsik",
    version="1.0.5",
    author="TGDK / Sean Tichenor",
    description="Direct Adaptive Trainer for Symbolic Instructional Knowledge",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "tqdm",
        "numpy",
    ],
    python_requires=">=3.9",
)
