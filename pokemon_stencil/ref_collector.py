"""
Automatic Pokémon reference image collector.

Downloads official artwork, sprites, and additional angles from public APIs
and stores them locally under ``refs/<pokemon_name>/``.  If fewer than
``min_images`` images are present in the directory after an initial fetch,
additional sprite variants are downloaded until the threshold is met or all
sources are exhausted.

Sources
-------
1. **PokéAPI** (``https://pokeapi.co/api/v2/pokemon/<name>``) – official HD
   artwork from the ``other.official-artwork`` sprites field.
2. **PokeAPI CDN sprites** – front default, front shiny, back default, back
   shiny, and Home model variants for pose diversity.
3. **Inline sprite URLs** – any additional URLs embedded in the PokéAPI
   sprites payload that have not already been downloaded.

Usage::

    from pokemon_stencil.ref_collector import ReferenceCollector

    collector = ReferenceCollector()
    paths = collector.collect("pikachu")          # → refs/pikachu/*.png
    paths = collector.collect("charizard", min_images=20)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_POKEAPI_BASE: str = "https://pokeapi.co/api/v2/pokemon"

#: Canonical URL for official HD artwork (transparent PNG, ~475 px).
_OFFICIAL_ART_TEMPLATE: str = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/other/official-artwork/{pokemon_id}.png"
)

#: Additional sprite URL templates ordered by quality / pose diversity.
_SPRITE_TEMPLATES: list[tuple[str, str]] = [
    (
        "front_shiny",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/shiny/{pokemon_id}.png",
    ),
    (
        "front_default",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/{pokemon_id}.png",
    ),
    (
        "back_default",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/back/{pokemon_id}.png",
    ),
    (
        "back_shiny",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/back/shiny/{pokemon_id}.png",
    ),
    (
        "home_front",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/other/home/{pokemon_id}.png",
    ),
    (
        "home_shiny",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/other/home/shiny/{pokemon_id}.png",
    ),
    (
        "showdown_front",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
        "sprites/pokemon/other/showdown/{pokemon_id}.gif",
    ),
]

#: Minimum image side length (px) – smaller images are discarded.
_MIN_SIZE: int = 96

#: Maximum normalised image side length (px).
_MAX_SIZE: int = 512

#: Default minimum reference image count before the collector stops fetching.
_DEFAULT_MIN_IMAGES: int = 15

#: Maximum images to keep per Pokémon.
_DEFAULT_MAX_IMAGES: int = 30

#: HTTP request timeout in seconds.
_HTTP_TIMEOUT: int = 15

#: Refs root directory.
_REFS_ROOT: Path = Path("refs")


def _lazy_requests():
    try:
        import requests  # type: ignore[import]
        return requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for reference collection. "
            "Install it with:  pip install requests"
        ) from exc


class ReferenceCollector:
    """
    Automatically downloads and stores Pokémon reference images.

    Args:
        refs_root: Root directory under which per-Pokémon sub-directories
            are created.  Defaults to ``refs/``.
        min_images: Minimum number of images to ensure are present.
            If the directory already has this many images the collector
            returns immediately without any network requests.
        max_images: Hard cap on the number of images stored per Pokémon.
        min_size: Minimum image side length in pixels; smaller images are
            discarded as too low quality for reference conditioning.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        refs_root: Path = _REFS_ROOT,
        min_images: int = _DEFAULT_MIN_IMAGES,
        max_images: int = _DEFAULT_MAX_IMAGES,
        min_size: int = _MIN_SIZE,
        timeout: int = _HTTP_TIMEOUT,
    ) -> None:
        self.refs_root = refs_root
        self.min_images = min_images
        self.max_images = max_images
        self.min_size = min_size
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    def collect(
        self,
        pokemon_name: str,
        min_images: Optional[int] = None,
    ) -> List[Path]:
        """
        Ensure at least *min_images* reference images exist for *pokemon_name*.

        Steps:

        1. Create ``refs/<pokemon_name>/`` if absent.
        2. Count existing images.  Return early if count >= *min_images*.
        3. Fetch Pokédex ID and image URLs from PokéAPI.
        4. Download images in priority order until *max_images* are saved
           or all sources are exhausted.
        5. If still fewer than *min_images* after the first pass, attempt
           additional sprite variants from the PokéAPI payload.

        Args:
            pokemon_name: Pokémon name (case-insensitive, spaces or hyphens).
            min_images: Override instance-level minimum.

        Returns:
            List of :class:`~pathlib.Path` objects for all images in the
            directory after collection (pre-existing + newly downloaded).
        """
        threshold = min_images if min_images is not None else self.min_images
        safe_name = pokemon_name.lower().replace(" ", "-")
        out_dir = self.refs_root / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = self._list_images(out_dir)
        if len(existing) >= threshold:
            logger.info(
                "Reference images for '%s': %d already present (min=%d). Skipping fetch.",
                safe_name, len(existing), threshold,
            )
            return existing

        logger.info(
            "Collecting reference images for '%s' (have %d, need %d).",
            safe_name, len(existing), threshold,
        )

        pokemon_id, urls = self._resolve_urls(safe_name)
        if not urls:
            logger.warning("No URLs resolved for '%s'. Cannot collect references.", safe_name)
            return existing

        already_saved = len(existing)
        saved: List[Path] = list(existing)

        for url in urls:
            if len(saved) >= self.max_images:
                break
            path = self._download(url, out_dir, index=len(saved))
            if path is not None:
                saved.append(path)

        # Second pass: extract additional URLs from the PokéAPI sprites tree.
        if len(saved) < threshold and pokemon_id is not None:
            extra_urls = self._fetch_extra_urls(safe_name)
            for url in extra_urls:
                if len(saved) >= self.max_images:
                    break
                if any(url in str(p) for p in saved):
                    continue
                path = self._download(url, out_dir, index=len(saved))
                if path is not None:
                    saved.append(path)

        newly = len(saved) - already_saved
        logger.info(
            "Reference collection for '%s' complete: %d image(s) downloaded, "
            "%d total (threshold=%d).",
            safe_name, newly, len(saved), threshold,
        )
        return saved

    def count(self, pokemon_name: str) -> int:
        """Return the number of existing reference images for *pokemon_name*."""
        safe_name = pokemon_name.lower().replace(" ", "-")
        out_dir = self.refs_root / safe_name
        return len(self._list_images(out_dir))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_urls(self, safe_name: str) -> tuple[Optional[int], List[str]]:
        """
        Query PokéAPI and build an ordered list of image URLs.

        Returns:
            ``(pokemon_id, [url, ...])`` ordered best-quality-first.
        """
        requests = _lazy_requests()
        api_url = f"{_POKEAPI_BASE}/{safe_name}"
        try:
            resp = requests.get(api_url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("PokéAPI request failed for '%s': %s", safe_name, exc)
            return None, []

        pokemon_id: int = data.get("id", 0)
        urls: List[str] = []

        # Official HD artwork first (best quality).
        urls.append(_OFFICIAL_ART_TEMPLATE.format(pokemon_id=pokemon_id))

        # Curated sprite variants.
        for _, template in _SPRITE_TEMPLATES:
            urls.append(template.format(pokemon_id=pokemon_id))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: List[str] = []
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                unique.append(url)

        return pokemon_id, unique

    def _fetch_extra_urls(self, safe_name: str) -> List[str]:
        """
        Return all image URLs from the PokéAPI sprites tree for *safe_name*.

        Used as a second-pass source when the primary URL list does not
        yield enough images.
        """
        requests = _lazy_requests()
        api_url = f"{_POKEAPI_BASE}/{safe_name}"
        try:
            resp = requests.get(api_url, timeout=self.timeout)
            resp.raise_for_status()
            sprites = resp.json().get("sprites", {})
        except Exception:
            return []
        return _extract_sprite_urls(sprites)

    def _download(
        self,
        url: str,
        out_dir: Path,
        index: int,
    ) -> Optional[Path]:
        """
        Download *url*, validate size, normalise, and save to *out_dir*.

        Returns the saved :class:`~pathlib.Path` or ``None`` on failure.
        """
        requests = _lazy_requests()
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        except Exception as exc:
            logger.debug("Download failed for %s: %s", url, exc)
            return None

        w, h = img.size
        if w < self.min_size or h < self.min_size:
            logger.debug("Discarding %s – size %dx%d below minimum %dpx.", url, w, h, self.min_size)
            return None

        # Composite transparent background.
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        # Downscale if oversized.
        if max(img.width, img.height) > _MAX_SIZE:
            scale = _MAX_SIZE / max(img.width, img.height)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )

        out_path = out_dir / f"ref_{index:03d}.png"
        img.save(out_path, format="PNG")
        logger.debug("Saved reference image: %s", out_path)
        return out_path

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        """Return sorted list of supported image paths in *directory*."""
        if not directory.is_dir():
            return []
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )


# ── Module-level convenience function ─────────────────────────────────────────

def collect_references(
    pokemon_name: str,
    min_images: int = _DEFAULT_MIN_IMAGES,
    refs_root: Path = _REFS_ROOT,
) -> List[Path]:
    """
    Convenience wrapper: collect reference images for *pokemon_name*.

    Equivalent to::

        ReferenceCollector(refs_root=refs_root, min_images=min_images)
            .collect(pokemon_name)

    Args:
        pokemon_name: Pokémon name (e.g. ``"pikachu"``).
        min_images: Minimum images to download if not already present.
        refs_root: Root directory for reference image storage.

    Returns:
        List of :class:`~pathlib.Path` objects for all images in the
        Pokémon's reference directory.
    """
    return ReferenceCollector(
        refs_root=refs_root,
        min_images=min_images,
    ).collect(pokemon_name)


# ── Internal utilities ─────────────────────────────────────────────────────────

def _extract_sprite_urls(sprites: dict) -> List[str]:
    """Recursively extract all HTTP URL strings from a nested sprites dict."""
    urls: List[str] = []
    for value in sprites.values():
        if isinstance(value, str) and value.startswith("http"):
            urls.append(value)
        elif isinstance(value, dict):
            urls.extend(_extract_sprite_urls(value))
    return urls
