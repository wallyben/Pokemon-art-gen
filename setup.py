"""Setup configuration for the Pokemon Stencil Art Factory."""

from setuptools import find_packages, setup

setup(
    name="pokemon-stencil-factory",
    version="1.0.0",
    description=(
        "Generate layered stencil SVG files from Pokemon artwork "
        "for Cricut cutting machines."
    ),
    author="Pokemon Art Gen",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        # Image processing
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        # Vector & SVG
        "svgwrite>=1.4.3",
        "pypotrace>=0.3",
        # CLI
        "click>=8.1.7",
        # Deep learning (CPU-compatible)
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.31.0",
        "accelerate>=0.21.0",
        "safetensors>=0.3.3",
        # ControlNet conditioning preprocessors
        "controlnet-aux>=0.0.7",
        # Model management / Hub downloads
        "huggingface_hub>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            # Installed as: pokemon-stencil
            "pokemon-stencil=pokemon_stencil.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
