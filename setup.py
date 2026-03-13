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
        "opencv-python>=4.8.0,<5.0.0",
        "numpy>=1.24.0,<2.0.0",
        "Pillow>=10.0.0,<11.0.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        # Vector & SVG (pypotrace removed: Windows-incompatible native build)
        "svgwrite>=1.4.3",
        # CLI & dashboard
        "click>=8.1.7",
        "streamlit>=1.28.0",
        # Deep learning — verified-compatible SDXL stack
        "torch>=2.1.0,<2.3.0",
        "diffusers>=0.27.0,<0.29.0",
        "transformers>=4.37.0,<4.40.0",
        "accelerate>=0.27.0,<0.29.0",
        "safetensors>=0.4.2,<0.5.0",
        # PEFT: required by diffusers>=0.27 for LoRA (load_lora_weights)
        "peft>=0.7.0,<0.9.0",
        # ControlNet conditioning preprocessors
        "controlnet-aux>=0.0.7,<0.1.0",
        # Model management / Hub downloads
        "huggingface_hub>=0.21.0,<0.23.0",
        # Reference image fetching
        "requests>=2.31.0",
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
