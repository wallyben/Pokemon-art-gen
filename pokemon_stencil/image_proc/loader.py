"""
Reference image loader.

Loads one or more reference images of a Pokemon from a directory or a list of
file paths, validates them, and returns consistent PIL Images ready for the
style-transfer and segmentation pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class ReferenceImageLoader:
    """
    Loads and validates reference images from disk.

    Usage::

        loader = ReferenceImageLoader(target_size=(512, 512))
        images = loader.load_from_directory("refs/pikachu/")
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        max_images: Optional[int] = None,
    ) -> None:
        """
        Args:
            target_size: (width, height) to which each image is resized.
            max_images: Optional cap on the number of images loaded.
        """
        self.target_size = target_size
        self.max_images = max_images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_directory(self, directory: Union[str, Path]) -> List[Image.Image]:
        """
        Load all supported images from *directory*.

        Args:
            directory: Path to a folder of reference images.

        Returns:
            List of PIL Images in RGB mode, resized to target_size.

        Raises:
            FileNotFoundError: If *directory* does not exist.
            ValueError: If no valid images are found.
        """
        dirpath = Path(directory)
        if not dirpath.is_dir():
            raise FileNotFoundError(f"Reference directory not found: {dirpath}")

        paths = sorted(
            p
            for p in dirpath.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not paths:
            raise ValueError(f"No supported images found in: {dirpath}")

        logger.info("Found %d reference image(s) in %s", len(paths), dirpath)
        return self.load_from_paths(paths)

    def load_from_paths(
        self, paths: List[Union[str, Path]]
    ) -> List[Image.Image]:
        """
        Load images from an explicit list of file paths.

        Args:
            paths: List of file paths to load.

        Returns:
            List of valid PIL Images (invalid files are skipped with a warning).
        """
        images: List[Image.Image] = []
        for raw_path in paths:
            path = Path(raw_path)
            img = self._safe_load(path)
            if img is not None:
                images.append(img)
            if self.max_images and len(images) >= self.max_images:
                break

        if not images:
            raise ValueError("No valid images could be loaded from the provided paths.")

        logger.info("Loaded %d valid reference image(s).", len(images))
        return images

    def load_single(self, path: Union[str, Path]) -> Image.Image:
        """
        Load a single image and raise on failure.

        Args:
            path: Path to the image file.

        Returns:
            PIL Image in RGB mode, resized to target_size.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = self._safe_load(p)
        if img is None:
            raise ValueError(f"Could not decode image: {p}")
        return img

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_load(self, path: Path) -> Optional[Image.Image]:
        """Attempt to load and normalise a single image; return None on error."""
        try:
            img = Image.open(path)
            img.verify()            # checks for corruption without decoding
            img = Image.open(path)  # re-open after verify (verify closes fp)
            img = self._normalise(img)
            logger.debug("Loaded: %s", path)
            return img
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            logger.warning("Skipping %s – could not load: %s", path, exc)
            return None

    def _normalise(self, img: Image.Image) -> Image.Image:
        """Convert to RGB and resize to target_size."""
        img = img.convert("RGB")
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.LANCZOS)
        return img

    @staticmethod
    def composite_references(images: List[Image.Image]) -> Image.Image:
        """
        Create a single representative image by averaging all references.

        This is a simple pixel-level blend that gives the generator a colour
        hint when multiple references are available.

        Args:
            images: List of same-size PIL Images in RGB.

        Returns:
            Blended PIL Image.
        """
        if not images:
            raise ValueError("Cannot composite an empty image list.")
        if len(images) == 1:
            return images[0]

        arrays = [np.array(img, dtype=np.float32) for img in images]
        averaged = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(averaged)
