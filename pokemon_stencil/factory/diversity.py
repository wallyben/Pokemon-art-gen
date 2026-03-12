"""
Design diversity filter for the art factory.

Prevents the factory from selecting multiple visually similar candidates
by computing structural fingerprints and enforcing a cosine-similarity
threshold during top-K selection.

Algorithm
---------
1. For each candidate image:
   a. Convert to grayscale.
   b. Run ``cv2.Canny`` edge detection to extract structural contours.
   c. Resize the edge map to 64 × 64 pixels.
   d. Flatten to a 4 096-element float32 vector.

2. Cosine similarity between any two fingerprints is:
   ``similarity = dot(A, B) / (||A|| × ||B||)``
   Range: 1.0 = identical structure, 0.0 = completely dissimilar.

3. Diverse selection (greedy):
   - Sort candidates by score, best first.
   - Accept a candidate only if its cosine similarity to **every**
     already-accepted candidate is below ``diversity_threshold``.
   - Stop once ``top_k`` diverse candidates are accumulated.

The first candidate is always accepted (no prior to compare against),
guaranteeing at least one output when valid candidates exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: Fingerprint side length in pixels (64 × 64 = 4 096 elements).
_FP_SIZE: int = 64


class DesignDiversityFilter:
    """
    Select a diverse set of candidate designs by structural fingerprint.

    Args:
        threshold: Cosine similarity threshold in ``[0, 1]``.  A candidate
            is rejected if its similarity to any already-selected design
            exceeds this value.  Default ``0.85`` rejects candidates that
            share more than 85 % structural similarity with an earlier pick.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"diversity_threshold must be in [0, 1], got {threshold!r}."
            )
        self.threshold = threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_fingerprint(self, image: Image.Image) -> np.ndarray:
        """
        Compute the structural fingerprint of *image*.

        Steps:

        1. Convert to grayscale.
        2. Apply Canny edge detection.
        3. Resize edge map to :data:`_FP_SIZE` × :data:`_FP_SIZE`.
        4. Flatten to a 1-D float32 vector of length ``_FP_SIZE ** 2``.

        Args:
            image: Source PIL Image (any mode).

        Returns:
            ``numpy.ndarray`` of shape ``(_FP_SIZE**2,)`` and dtype
            ``float32``.
        """
        gray = np.array(image.convert("L"), dtype=np.uint8)
        edges = cv2.Canny(gray, 50, 150)
        resized = cv2.resize(
            edges, (_FP_SIZE, _FP_SIZE), interpolation=cv2.INTER_AREA
        )
        return resized.flatten().astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two fingerprint vectors.

        Args:
            a: First vector (float32, shape ``(N,)``).
            b: Second vector (float32, shape ``(N,)``).

        Returns:
            Scalar similarity in ``[0.0, 1.0]``.  Returns ``0.0`` when
            either vector has zero magnitude (preventing division by zero).
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def select_diverse(
        self,
        candidates: "Sequence[_CandidateLike]",
        top_k: int,
    ) -> "List[_CandidateLike]":
        """
        Select up to *top_k* structurally diverse candidates.

        Candidates are evaluated in descending score order.  A candidate
        is accepted if and only if its cosine similarity to every previously
        accepted candidate is strictly below :attr:`threshold`.

        Candidates without a valid ``simplified_path`` (i.e. failed
        generation attempts) are silently skipped.

        Args:
            candidates: Sequence of :class:`CandidateResult`-like objects
                that expose ``score: float`` and
                ``simplified_path: Optional[Path]`` attributes.
            top_k: Maximum number of candidates to return.

        Returns:
            List of accepted candidates, ordered by descending score.
        """
        sorted_candidates = sorted(
            candidates, key=lambda c: c.score, reverse=True
        )

        selected: "List[_CandidateLike]" = []
        selected_fingerprints: List[np.ndarray] = []

        for candidate in sorted_candidates:
            if len(selected) >= top_k:
                break

            if candidate.simplified_path is None:
                logger.debug(
                    "Skipping candidate %s – no simplified_path.",
                    getattr(candidate, "index", "?"),
                )
                continue

            try:
                img = Image.open(candidate.simplified_path).convert("RGB")
                fp = self.compute_fingerprint(img)
            except Exception as exc:
                logger.warning(
                    "Could not compute fingerprint for candidate %s: %s",
                    getattr(candidate, "index", "?"),
                    exc,
                )
                continue

            if _is_diverse(fp, selected_fingerprints, self.threshold):
                selected.append(candidate)
                selected_fingerprints.append(fp)
                logger.debug(
                    "Accepted candidate %s (score=%.3f, selected=%d/%d).",
                    getattr(candidate, "index", "?"),
                    candidate.score,
                    len(selected),
                    top_k,
                )
            else:
                logger.debug(
                    "Rejected candidate %s – too similar to an existing selection.",
                    getattr(candidate, "index", "?"),
                )

        logger.info(
            "Diversity filter: selected %d/%d candidates (threshold=%.2f).",
            len(selected),
            len(sorted_candidates),
            self.threshold,
        )
        return selected


# ── Private helpers ───────────────────────────────────────────────────────────

def _is_diverse(
    fp: np.ndarray,
    existing: List[np.ndarray],
    threshold: float,
) -> bool:
    """Return ``True`` iff *fp* is below *threshold* similarity to all *existing*."""
    for existing_fp in existing:
        norm_a = float(np.linalg.norm(fp))
        norm_b = float(np.linalg.norm(existing_fp))
        if norm_a == 0.0 or norm_b == 0.0:
            continue
        sim = float(np.dot(fp, existing_fp) / (norm_a * norm_b))
        if sim >= threshold:
            return False
    return True


# Structural type alias for duck-typing (avoids circular import).
class _CandidateLike:
    """Protocol-like type hint; not enforced at runtime."""

    score: float
    simplified_path: Optional[Path]
