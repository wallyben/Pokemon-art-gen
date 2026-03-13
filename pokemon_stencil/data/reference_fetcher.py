"""
Automatic reference image acquisition for Pokémon artwork.

Fetches high-quality official artwork and sprite images from public sources,
filters out low-resolution results, deduplicates using perceptual hashing,
and stores the images locally for use by the generation pipeline.

Sources used
------------
1. **PokéAPI** (``https://pokeapi.co/api/v2/pokemon/<name>``) – resolves the
   numeric Pokédex ID and official artwork URL via the
   ``sprites.other.official-artwork.front_default`` field.
2. **PokeAPI sprite CDN** – additional front/back default sprites for pose
   variety.

Duplicate detection uses ``imagehash`` perceptual hashing when available;
falls back to file-size–based deduplication otherwise.

Usage
-----
::

    from pokemon_stencil.data.reference_fetcher import ReferenceImageFetcher
    from pathlib import Path

    fetcher = ReferenceImageFetcher(max_images=10)
    saved = fetcher.fetch_references("pikachu", Path("refs/pikachu"))

    # Or prepare an existing directory (dedup + normalise):
    fetcher.prepare_reference_images(Path("refs/pikachu"))
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

from PIL import Image

logger = logging.getLogger(__name__)

#: Minimum side length (px) for an image to be considered usable.
_MIN_RESOLUTION: int = 256

#: Maximum side length after normalisation (px); larger images are downscaled.
_MAX_NORMALISED_SIZE: int = 512

#: PokeAPI base URL.
_POKEAPI_BASE: str = "https://pokeapi.co/api/v2/pokemon"

#: Canonical URL pattern for official HD artwork on the PokeAPI CDN.
_OFFICIAL_ART_URL: str = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/other/official-artwork/{pokemon_id}.png"
)

#: Additional sprite URL templates keyed by variant name.
_SPRITE_URL_TEMPLATES: dict[str, str] = {
    "front_default": (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/{pokemon_id}.png"
    ),
    "front_shiny": (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/shiny/{pokemon_id}.png"
    ),
    "front_home": (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/other/home/{pokemon_id}.png"
    ),
}

#: Perceptual hash difference threshold for considering two images duplicates.
#: Smaller = stricter (0 = identical, 10 = very similar).
_PHASH_THRESHOLD: int = 10


def _import_requests():
    """Lazy import for ``requests`` to keep startup cost low."""
    try:
        import requests  # type: ignore[import]
        return requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for reference fetching. "
            "Install it with: pip install requests"
        ) from exc


def _import_imagehash():
    """Lazy import for ``imagehash``.  Returns ``None`` if not installed."""
    try:
        import imagehash  # type: ignore[import]
        return imagehash
    except ImportError:
        return None


class ReferenceImageFetcher:
    """
    Download and prepare Pokémon reference images from public sources.

    Args:
        max_images: Maximum number of images to save per Pokémon.
            Default is 25.
        min_resolution: Minimum image side length in pixels.  Images
            smaller than this in either dimension are discarded.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        max_images: int = 25,
        min_resolution: int = _MIN_RESOLUTION,
        timeout: int = 15,
    ) -> None:
        self.max_images = max_images
        self.min_resolution = min_resolution
        self.timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_references(
        self,
        pokemon_name: str,
        output_dir: Path,
    ) -> List[Path]:
        """
        Fetch reference images for *pokemon_name* and save to *output_dir*.

        Steps:

        1. Query PokéAPI to resolve the Pokédex ID and collect image URLs.
        2. Download each image.
        3. Discard images below :attr:`min_resolution` in either dimension.
        4. Resize images exceeding :data:`_MAX_NORMALISED_SIZE`.
        5. Convert to RGB and save as PNG.
        6. Stop once :attr:`max_images` images are saved.

        Args:
            pokemon_name: Canonical Pokémon name (case-insensitive, e.g.
                ``"Pikachu"`` or ``"mr-mime"``).
            output_dir: Directory to write image files into; created if absent.

        Returns:
            List of :class:`~pathlib.Path` objects for successfully saved
            files.  An empty list is returned when no usable images were
            found or all network requests failed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = pokemon_name.lower().replace(" ", "-")

        logger.info(
            "Fetching references | pokemon=%s | output_dir=%s | max=%d",
            safe_name,
            output_dir,
            self.max_images,
        )

        # ── Step 1: Resolve Pokédex ID and collect URLs ───────────────────────
        pokemon_id, image_urls = self._resolve_pokemon(safe_name)
        if not image_urls:
            logger.warning(
                "No image URLs found for '%s'. "
                "Check the Pokémon name spelling.",
                safe_name,
            )
            return []

        # ── Steps 2-6: Download, filter, and save ─────────────────────────────
        saved: List[Path] = []
        for url in image_urls:
            if len(saved) >= self.max_images:
                break
            path = self._download_and_save(url, output_dir, len(saved))
            if path is not None:
                saved.append(path)

        logger.info(
            "Fetched %d/%d reference image(s) for '%s'.",
            len(saved),
            self.max_images,
            safe_name,
        )
        return saved

    def prepare_reference_images(self, image_dir: Path) -> List[Path]:
        """
        Post-process a directory of reference images in-place.

        Operations applied:

        1. **Duplicate removal** – perceptual hash comparison
           (``imagehash`` library) or file-size fallback.
        2. **Size normalisation** – images larger than
           :data:`_MAX_NORMALISED_SIZE` in either dimension are downscaled.
        3. **Transparent background removal** – RGBA images are composited
           onto a white RGB background.
        4. Files that cannot be read are silently skipped.

        Args:
            image_dir: Directory containing PNG/JPG reference images.

        Returns:
            List of :class:`~pathlib.Path` objects for files that remain
            after preparation.
        """
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

        if not image_paths:
            logger.debug("prepare_reference_images: no images found in %s.", image_dir)
            return []

        imagehash = _import_imagehash()
        kept: List[Path] = []
        seen_hashes: list = []
        seen_sizes: list[int] = []

        for path in image_paths:
            try:
                img = Image.open(path)
            except Exception as exc:
                logger.warning("Cannot open %s: %s – skipping.", path.name, exc)
                continue

            # ── Deduplication ─────────────────────────────────────────────────
            if imagehash is not None:
                try:
                    phash = imagehash.phash(img)
                    if any(
                        abs(phash - existing) <= _PHASH_THRESHOLD
                        for existing in seen_hashes
                    ):
                        logger.debug("Removing duplicate: %s", path.name)
                        path.unlink(missing_ok=True)
                        continue
                    seen_hashes.append(phash)
                except Exception:
                    pass  # Fall through to size-based dedup.
            else:
                file_size = path.stat().st_size
                if file_size in seen_sizes:
                    logger.debug(
                        "Removing size-duplicate: %s", path.name
                    )
                    path.unlink(missing_ok=True)
                    continue
                seen_sizes.append(file_size)

            # ── Transparent background removal ────────────────────────────────
            img = _remove_transparent_background(img)

            # ── Size normalisation ────────────────────────────────────────────
            img = _normalise_size(img, _MAX_NORMALISED_SIZE)

            # Write back.
            img.save(path, format="PNG")
            kept.append(path)

        logger.info(
            "prepare_reference_images: %d/%d images kept in %s.",
            len(kept),
            len(image_paths),
            image_dir,
        )
        return kept

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_pokemon(
        self, safe_name: str
    ) -> tuple[Optional[int], List[str]]:
        """
        Query the PokéAPI and collect usable image URLs.

        Args:
            safe_name: URL-safe Pokémon name (lowercase, hyphens).

        Returns:
            ``(pokemon_id, [url, ...])`` where *pokemon_id* may be ``None``
            on failure and the URL list is ordered best-first (official art
            before sprites).
        """
        requests = _import_requests()
        api_url = f"{_POKEAPI_BASE}/{safe_name}"
        try:
            response = requests.get(api_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("PokéAPI request failed for '%s': %s", safe_name, exc)
            return None, []

        pokemon_id: int = data.get("id", 0)
        urls: List[str] = []

        # Prefer official HD artwork first.
        official_art_url = _OFFICIAL_ART_URL.format(pokemon_id=pokemon_id)
        urls.append(official_art_url)

        # Add sprite variants.
        for _, template in _SPRITE_URL_TEMPLATES.items():
            urls.append(template.format(pokemon_id=pokemon_id))

        # Extract any URLs directly from the API sprites payload.
        sprites = data.get("sprites", {})
        urls.extend(_extract_sprite_urls(sprites))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_urls: List[str] = []
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return pokemon_id, unique_urls

    def _download_and_save(
        self,
        url: str,
        output_dir: Path,
        index: int,
    ) -> Optional[Path]:
        """
        Download *url*, validate resolution, and save to *output_dir*.

        Args:
            url: Image URL to download.
            output_dir: Target directory.
            index: Zero-based index used to name the file.

        Returns:
            :class:`~pathlib.Path` of the saved file, or ``None`` on any
            failure (network error, resolution too small, etc.).
        """
        requests = _import_requests()
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        except Exception as exc:
            logger.debug("Failed to download %s: %s", url, exc)
            return None

        # Resolution filter.
        w, h = img.size
        if w < self.min_resolution or h < self.min_resolution:
            logger.debug(
                "Discarding %s – resolution %dx%d below minimum %dpx.",
                url, w, h, self.min_resolution,
            )
            return None

        # Normalise.
        img = _remove_transparent_background(img)
        img = _normalise_size(img, _MAX_NORMALISED_SIZE)

        out_path = output_dir / f"ref_{index:03d}.png"
        img.save(out_path, format="PNG")
        logger.debug("Saved reference image: %s", out_path)
        return out_path


# ── Module-level helpers ──────────────────────────────────────────────────────

def _remove_transparent_background(img: Image.Image) -> Image.Image:
    """Composite RGBA images onto a white RGB background."""
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")


def _normalise_size(img: Image.Image, max_size: int) -> Image.Image:
    """Downscale *img* so its longest side is at most *max_size* px."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _extract_sprite_urls(sprites: dict) -> List[str]:
    """
    Recursively extract all non-null URL strings from a sprites dict.

    Args:
        sprites: Nested dict from the PokéAPI ``sprites`` field.

    Returns:
        Flat list of URL strings.
    """
    urls: List[str] = []
    for value in sprites.values():
        if isinstance(value, str) and value.startswith("http"):
            urls.append(value)
        elif isinstance(value, dict):
            urls.extend(_extract_sprite_urls(value))
    return urls
