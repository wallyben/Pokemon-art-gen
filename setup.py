"""Setup configuration for Pokemon Stencil Art Factory."""

from setuptools import setup, find_packages

setup(
    name="pokemon-stencil-factory",
    version="1.0.0",
    description="Generate layered stencil SVGs from Pokemon artwork for Cricut cutting machines.",
    author="Pokemon Art Gen",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "svgwrite>=1.4.3",
        "pypotrace>=0.3",
        "scikit-learn>=1.3.0",
        "click>=8.1.7",
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.31.0",
        "accelerate>=0.21.0",
        "scipy>=1.11.0",
    ],
    entry_points={
        "console_scripts": [
            "pokemon-stencil=pokemon_stencil.cli:main",
        ],
    },
)
