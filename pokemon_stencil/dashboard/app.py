"""
Pokémon Stencil Art Factory — Streamlit dashboard (v3 – SDXL upgrade).

Upgraded for the SDXL + IP-Adapter + ControlNet pipeline:
- Model selection (SDXL / legacy DreamShaper)
- Reference images preview
- Pose reference image upload
- LoRA model selection from models/lora/
- Generation seed control
- Prompt optimisation toggle
- IP-Adapter toggle
- Candidate preview grid before export
- SVG download per layer

Usage (invoked by the CLI)::

    pokemon-stencil dashboard          # opens http://localhost:8501
    pokemon-stencil dashboard --port 8502

Layout
------
Sidebar
    • Pokémon name (text input)
    • Description / pose prompt (textarea)
    • Model selection (SDXL / DreamShaper legacy)
    • IP-Adapter toggle
    • Prompt optimisation toggle
    • Generation seed
    • Candidate count slider (1–20)
    • Top results slider (1–5)
    • LoRA selection dropdown
    • Pose reference image upload
    • Auto-fetch references toggle
    • Output directory

Main panel
    • Stage-by-stage progress bar + status label
    • Per-candidate live progress during generation
    • Reference images preview grid
    • Candidate preview grid (all generated candidates)
    • Generation results grid with SVG downloads
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, List, Optional

import streamlit as st

# ── Page config must be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Pokémon Stencil Art Factory",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pokemon_stencil.config import (  # noqa: E402  (after set_page_config)
    DEFAULT_DREAMSHAPER_HUB_ID,
    DEFAULT_LORA_DIR,
    DEFAULT_MODEL_HUB_ID,
    DEFAULT_SDXL_HUB_ID,
    FactoryConfig,
    GenerationConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from pokemon_stencil.factory.factory_runner import FactoryResult, FactoryRunner
from pokemon_stencil.models.model_manager import ModelManager
from pokemon_stencil.pipeline import PipelineResult

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_LABELS: list[str] = [
    "Fetching references",
    "Generating composition guidance",
    "Generating candidates",
    "Scoring & diversity filtering",
    "Converting to stencil SVGs",
]

_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_REFS_DIR = Path("refs")

_MODEL_OPTIONS = {
    "SDXL 1.0 (recommended – best quality)": DEFAULT_SDXL_HUB_ID,
    "DreamShaper SD1.5 (faster, lower VRAM)": DEFAULT_DREAMSHAPER_HUB_ID,
    "Stable Diffusion 1.5 (legacy)": DEFAULT_MODEL_HUB_ID,
}

_LORA_NONE_LABEL = "None (no LoRA)"


# ─────────────────────────────────────────────────────────────────────────────
# Progress-aware factory runner
# ─────────────────────────────────────────────────────────────────────────────

class _ProgressFactoryRunner(FactoryRunner):
    """FactoryRunner subclass that fires progress callbacks at key milestones."""

    def __init__(
        self,
        config: PipelineConfig,
        on_candidate_done: Optional[Callable[[int, int], None]] = None,
        on_stencil_done: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        super().__init__(config)
        self._on_candidate_done = on_candidate_done
        self._on_stencil_done = on_stencil_done
        self._total_candidates: int = 0
        self._done_candidates: int = 0
        self._total_stencils: int = 0
        self._done_stencils: int = 0

    def _run_sequential(self, pokemon_name, count, reference_dir, prompt_extra):
        self._total_candidates = count
        self._done_candidates = 0
        return super()._run_sequential(pokemon_name, count, reference_dir, prompt_extra)

    def _generate_one(
        self,
        pokemon_name,
        candidate_index,
        reference_dir,
        prompt_extra,
        composition_map=None,
    ):
        result = super()._generate_one(
            pokemon_name, candidate_index, reference_dir, prompt_extra, composition_map
        )
        self._done_candidates += 1
        if self._on_candidate_done:
            self._on_candidate_done(self._done_candidates, self._total_candidates)
        return result

    def _run_stencil_on_candidate(self, candidate, run_name):
        result = super()._run_stencil_on_candidate(candidate, run_name)
        self._done_stencils += 1
        if self._on_stencil_done:
            self._on_stencil_done(self._done_stencils, self._total_stencils)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_available_loras() -> List[str]:
    """Return available LoRA names from the default lora directory."""
    manager = ModelManager()
    names = manager.lora_names()
    return [_LORA_NONE_LABEL] + names


def _build_config(
    output_dir: Path,
    candidate_count: int,
    top_k: int,
    model_hub_id: str,
    seed: Optional[int],
    optimise_prompt: bool,
    use_controlnet: bool,
    use_ip_adapter: bool,
    lora_name: Optional[str],
) -> PipelineConfig:
    """Construct a PipelineConfig for dashboard use."""
    lora_path: Optional[Path] = None
    if lora_name and lora_name != _LORA_NONE_LABEL:
        candidate = DEFAULT_LORA_DIR / f"{lora_name}.safetensors"
        if candidate.is_file():
            lora_path = candidate

    # Detect if SDXL model is selected.
    is_sdxl = model_hub_id == DEFAULT_SDXL_HUB_ID

    gen_config = GenerationConfig(
        num_inference_steps=28 if is_sdxl else 20,
        guidance_scale=7.5,
        width=1024 if is_sdxl else 768,
        height=1024 if is_sdxl else 768,
        device="cpu",
        torch_dtype="float32",
        seed=seed,
        optimise_prompt=optimise_prompt,
        use_controlnet=use_controlnet,
        use_ip_adapter=use_ip_adapter and is_sdxl,
        lora_path=lora_path,
    )

    # Point legacy model paths for non-SDXL selections.
    if not is_sdxl:
        gen_config.legacy_model_hub_id = model_hub_id

    return PipelineConfig(
        generation=gen_config,
        processing=ProcessingConfig(
            output_size=(gen_config.width, gen_config.height),
        ),
        output=OutputConfig(base_dir=output_dir),
        factory=FactoryConfig(
            count=candidate_count,
            top_k=top_k,
            workers=1,
        ),
        skip_generation=False,
    )


def _find_preview_images(output_dir: Path, run_name: str) -> List[Path]:
    """Return source images for *run_name*, newest first."""
    images_dir = output_dir / run_name / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def _find_candidate_images(output_dir: Path) -> List[Path]:
    """Return all candidate simplified images from the factory candidates dir."""
    candidates_dir = output_dir / "_factory_candidates"
    if not candidates_dir.is_dir():
        return []
    return sorted(candidates_dir.glob("candidate_*.png"))


def _find_svg_files(output_dir: Path, run_name: str) -> List[Path]:
    """Return all SVG stencil files produced for *run_name*."""
    svg_dir = output_dir / run_name / "svg"
    if not svg_dir.is_dir():
        return []
    return sorted(svg_dir.glob("*.svg"))


def _refs_dir_for(pokemon_name: str) -> Path:
    """Return the conventional refs directory for *pokemon_name*."""
    safe = pokemon_name.lower().replace(" ", "-")
    return _DEFAULT_REFS_DIR / safe


def _find_ref_images(refs_dir: Path) -> List[Path]:
    """Return all reference image paths in *refs_dir*."""
    if not refs_dir.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in refs_dir.iterdir() if p.suffix.lower() in suffixes)


# ─────────────────────────────────────────────────────────────────────────────
# UI sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    """Render all input widgets in the sidebar."""
    st.sidebar.title("🎨 Stencil Factory")
    st.sidebar.markdown(
        "Generate Cricut-ready stencil SVGs from a text prompt.\n\n"
        "**v3 — SDXL + IP-Adapter + ControlNet**"
    )
    st.sidebar.divider()

    pokemon_name = st.sidebar.text_input(
        "Pokémon name",
        value="Pikachu",
        help="Case-insensitive. Use hyphens for multi-word names (e.g. mr-mime).",
    )

    prompt = st.sidebar.text_area(
        "Description / pose",
        value="dynamic pose, bold outline",
        height=90,
        help="Describe the desired pose or environment. Appended to the generation prompt.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Model Settings")

    model_label = st.sidebar.selectbox(
        "Base model",
        options=list(_MODEL_OPTIONS.keys()),
        index=0,
        help=(
            "SDXL 1.0 produces the highest quality output at 1024×1024. "
            "DreamShaper is faster on lower-VRAM machines."
        ),
    )
    model_hub_id = _MODEL_OPTIONS[model_label]
    is_sdxl = model_hub_id == DEFAULT_SDXL_HUB_ID

    use_controlnet = st.sidebar.toggle(
        "Enable ControlNet (OpenPose + Canny)",
        value=True,
        help="Dual ControlNet conditioning for precise pose and structure guidance.",
    )

    use_ip_adapter = st.sidebar.toggle(
        "Enable IP-Adapter (reference conditioning)",
        value=is_sdxl,
        disabled=not is_sdxl,
        help=(
            "Encodes Pokémon reference images via CLIP to preserve character anatomy. "
            "Requires SDXL model and reference images in refs/<name>/."
        ),
    )

    optimise_prompt = st.sidebar.toggle(
        "Optimise prompt (77-token limit)",
        value=True,
        help="Automatically compress prompts to fit within the CLIP 77-token limit.",
    )

    seed_input = st.sidebar.number_input(
        "Generation seed (-1 = random)",
        min_value=-1,
        max_value=2**31 - 1,
        value=-1,
        step=1,
        help="Set a fixed seed for reproducible results. -1 uses a random seed.",
    )
    seed: Optional[int] = None if seed_input == -1 else int(seed_input)

    st.sidebar.divider()
    st.sidebar.subheader("Generation Settings")

    count = st.sidebar.slider(
        "Candidate designs",
        min_value=1,
        max_value=20,
        value=6,
        help=(
            "Number of artwork candidates to generate before scoring. "
            "20 candidates per spec for best results."
        ),
    )

    top_k = st.sidebar.slider(
        "Top results to export",
        min_value=1,
        max_value=min(5, count),
        value=min(2, count),
        help="Best-scoring, most diverse designs converted to SVG stencil packs.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("LoRA & References")

    # ── LoRA selection ─────────────────────────────────────────────────────────
    available_loras = _get_available_loras()
    lora_label = st.sidebar.selectbox(
        "Character LoRA",
        options=available_loras,
        index=0,
        help=(
            "Select a character-specific LoRA from models/lora/. "
            "Place .safetensors files in that directory to enable this option. "
            "LoRA weights are merged into the pipeline for character refinement."
        ),
    )
    lora_name: Optional[str] = (
        None if lora_label == _LORA_NONE_LABEL else lora_label
    )

    if lora_name:
        st.sidebar.success(f"LoRA: `{lora_name}.safetensors`")
    elif len(available_loras) == 1:
        st.sidebar.caption(
            "No LoRA files found. Place `.safetensors` files in `models/lora/` "
            "to enable character-specific refinement."
        )

    pose_reference = st.sidebar.file_uploader(
        "Pose reference image (optional)",
        type=["png", "jpg", "jpeg"],
        help=(
            "Upload a pose reference image. OpenPose will extract the skeleton "
            "and use it as ControlNet conditioning."
        ),
    )

    auto_fetch = st.sidebar.toggle(
        "Auto-fetch reference images",
        value=False,
        help=(
            "Download official Pokémon artwork before generation to improve "
            "IP-Adapter conditioning. Requires internet access."
        ),
    )

    output_dir = st.sidebar.text_input(
        "Output directory",
        value="outputs",
        help="Root directory for all generated files.",
    )

    return dict(
        pokemon_name=pokemon_name,
        prompt=prompt,
        count=count,
        top_k=top_k,
        auto_fetch=auto_fetch,
        output_dir=Path(output_dir),
        model_hub_id=model_hub_id,
        model_label=model_label,
        is_sdxl=is_sdxl,
        use_controlnet=use_controlnet,
        use_ip_adapter=use_ip_adapter,
        optimise_prompt=optimise_prompt,
        seed=seed,
        pose_reference=pose_reference,
        lora_name=lora_name,
    )


def _render_reference_preview(pokemon_name: str) -> None:
    """Display reference images for *pokemon_name* if they exist on disk."""
    refs_dir = _refs_dir_for(pokemon_name)
    ref_paths = _find_ref_images(refs_dir)
    if not ref_paths:
        return

    with st.expander(
        f"📸 Reference images ({len(ref_paths)} found in `{refs_dir}`)",
        expanded=False,
    ):
        cols = st.columns(min(len(ref_paths), 6))
        for i, ref_path in enumerate(ref_paths[:6]):
            with cols[i % 6]:
                st.image(str(ref_path), caption=ref_path.name, use_container_width=True)
        if len(ref_paths) > 6:
            st.caption(f"… and {len(ref_paths) - 6} more reference images.")


def _render_candidate_preview_grid(output_dir: Path) -> None:
    """
    Display a preview grid of all generated candidates before export.

    Shows all simplified candidate images from the factory candidates directory.
    """
    candidate_paths = _find_candidate_images(output_dir)
    if not candidate_paths:
        return

    with st.expander(
        f"🖼 Candidate preview grid ({len(candidate_paths)} candidates)",
        expanded=True,
    ):
        st.caption(
            "All generated candidates shown below. "
            "Top-scoring, most diverse designs are exported as SVG stencil packs."
        )
        # Show in a grid of up to 5 columns.
        n_cols = min(len(candidate_paths), 5)
        cols = st.columns(n_cols)
        for i, img_path in enumerate(candidate_paths):
            with cols[i % n_cols]:
                st.image(
                    str(img_path),
                    caption=f"Candidate {i:02d}",
                    use_container_width=True,
                )


def _render_progress_area() -> dict:
    """Create and return Streamlit placeholder containers for the progress UI."""
    st.subheader("⚙️ Progress")
    stage_label = st.empty()
    stage_bar = st.progress(0, text="Waiting…")
    st.markdown("")
    candidate_label = st.empty()
    candidate_bar = st.progress(0)
    log = st.empty()
    return dict(
        stage_bar=stage_bar,
        stage_label=stage_label,
        candidate_bar=candidate_bar,
        candidate_label=candidate_label,
        log=log,
    )


def _render_results(
    factory_result: FactoryResult,
    output_dir: Path,
) -> None:
    """Display candidate preview grid and SVG download buttons for each design."""
    st.subheader("✅ Results")

    if not factory_result.ranked:
        st.info("No stencil packs were produced. Try increasing the candidate count.")
        return

    # ── Candidate preview grid ─────────────────────────────────────────────────
    _render_candidate_preview_grid(output_dir)

    # ── Per-design results ─────────────────────────────────────────────────────
    for score, run_name in factory_result.ranked:
        with st.expander(f"🖼  {run_name}  —  score {score:.3f}", expanded=True):
            preview_images = _find_preview_images(output_dir, run_name)
            svg_files = _find_svg_files(output_dir, run_name)

            col_img, col_dl = st.columns([3, 1])

            with col_img:
                if preview_images:
                    grid_images = preview_images[:4]
                    if len(grid_images) == 1:
                        st.image(
                            str(grid_images[0]),
                            caption=f"{run_name} — source image",
                            use_container_width=True,
                        )
                    else:
                        grid_cols = st.columns(len(grid_images))
                        for j, img_path in enumerate(grid_images):
                            with grid_cols[j]:
                                st.image(
                                    str(img_path),
                                    caption=f"img {j+1}",
                                    use_container_width=True,
                                )
                else:
                    st.caption("No preview image available.")

            with col_dl:
                st.markdown(f"**Score:** `{score:.3f}`")
                if svg_files:
                    st.markdown(f"**SVG layers:** {len(svg_files)}")
                    for svg_path in svg_files:
                        svg_bytes = svg_path.read_bytes()
                        st.download_button(
                            label=f"⬇ {svg_path.name}",
                            data=svg_bytes,
                            file_name=svg_path.name,
                            mime="image/svg+xml",
                            key=f"dl_{run_name}_{svg_path.stem}",
                        )
                else:
                    st.caption("No SVG files found.")

    if factory_result.errors:
        with st.expander(f"⚠️ {len(factory_result.errors)} error(s)", expanded=False):
            for err in factory_result.errors:
                st.code(err[:500], language="text")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for ``streamlit run`` or the CLI launcher."""

    inputs = _render_sidebar()

    st.title("Pokémon Stencil Art Factory")
    st.markdown(
        "Generate stencil-ready SVG artwork for Cricut cutters using the "
        "**SDXL + IP-Adapter + ControlNet** pipeline. "
        "Configure your design in the sidebar, then press **Generate Artwork**."
    )

    if not inputs["pokemon_name"].strip():
        st.warning("Enter a Pokémon name in the sidebar to get started.")
        return

    # ── Model info banner ─────────────────────────────────────────────────────
    controlnet_status = "✅ ControlNet on" if inputs["use_controlnet"] else "⚠️ ControlNet off"
    ipa_status = "✅ IP-Adapter on" if inputs["use_ip_adapter"] else "⚠️ IP-Adapter off"
    lora_status = f"LoRA: `{inputs['lora_name']}`" if inputs["lora_name"] else "No LoRA"
    st.info(
        f"**Model:** {inputs['model_label']}  |  {controlnet_status}  |  "
        f"{ipa_status}  |  **Prompt opt:** {'✅' if inputs['optimise_prompt'] else '⚠️'}  |  "
        f"**Seed:** {inputs['seed'] if inputs['seed'] is not None else 'random'}  |  "
        f"{lora_status}"
    )

    _render_reference_preview(inputs["pokemon_name"])

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_output_dir" not in st.session_state:
        st.session_state.last_output_dir = None

    generate_clicked = st.button(
        "🚀 Generate Artwork",
        type="primary",
        use_container_width=True,
    )

    if not generate_clicked and st.session_state.last_result is not None:
        _render_results(
            st.session_state.last_result,
            st.session_state.last_output_dir,
        )
        return

    if not generate_clicked:
        st.info(
            "Configure your settings in the sidebar, then click "
            "**Generate Artwork** to start."
        )
        return

    # ── Run generation ────────────────────────────────────────────────────────
    pokemon_name: str = inputs["pokemon_name"].strip()
    prompt: str = inputs["prompt"].strip()
    count: int = inputs["count"]
    top_k: int = inputs["top_k"]
    output_dir: Path = inputs["output_dir"]
    auto_fetch: bool = inputs["auto_fetch"]

    progress_area = _render_progress_area()
    stage_bar = progress_area["stage_bar"]
    stage_label = progress_area["stage_label"]
    candidate_bar = progress_area["candidate_bar"]
    candidate_label = progress_area["candidate_label"]
    log_placeholder = progress_area["log"]

    def _set_stage(idx: int, label: str) -> None:
        frac = idx / len(_STAGE_LABELS)
        stage_bar.progress(frac, text=f"Stage {idx}/{len(_STAGE_LABELS)}: {label}")
        stage_label.markdown(f"**Current stage:** {label}")

    def _on_candidate_done(done: int, total: int) -> None:
        frac = done / max(total, 1)
        candidate_bar.progress(frac)
        candidate_label.markdown(f"Candidate **{done}** / {total} generated")

    def _on_stencil_done(done: int, total: int) -> None:
        frac = done / max(total, 1)
        candidate_bar.progress(frac)
        candidate_label.markdown(f"Stencil pack **{done}** / {total} exported")

    try:
        _set_stage(1, _STAGE_LABELS[0])
        refs_dir: Optional[Path] = None
        if auto_fetch:
            refs_dir = _refs_dir_for(pokemon_name)
            refs_dir.mkdir(parents=True, exist_ok=True)
            log_placeholder.info(f"Reference directory: `{refs_dir}`")

        # Save uploaded pose reference to temp file.
        pose_ref_path: Optional[Path] = None
        if inputs["pose_reference"] is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(inputs["pose_reference"].getvalue())
                pose_ref_path = Path(tmp.name)
            log_placeholder.info(f"Pose reference saved to `{pose_ref_path}`")

        config = _build_config(
            output_dir=output_dir,
            candidate_count=count,
            top_k=top_k,
            model_hub_id=inputs["model_hub_id"],
            seed=inputs["seed"],
            optimise_prompt=inputs["optimise_prompt"],
            use_controlnet=inputs["use_controlnet"],
            use_ip_adapter=inputs["use_ip_adapter"],
            lora_name=inputs["lora_name"],
        )
        config.generation.auto_fetch_references = auto_fetch
        config.generation.max_reference_images = 25
        if pose_ref_path:
            config.generation.pose_reference_path = pose_ref_path

        _set_stage(2, _STAGE_LABELS[1])
        _set_stage(3, _STAGE_LABELS[2])
        candidate_label.markdown(f"Generating {count} candidate(s)…")
        candidate_bar.progress(0)

        runner = _ProgressFactoryRunner(
            config,
            on_candidate_done=_on_candidate_done,
            on_stencil_done=_on_stencil_done,
        )
        runner._total_stencils = top_k

        result: FactoryResult = runner.run(
            pokemon_name=pokemon_name,
            reference_dir=refs_dir,
            prompt_extra=prompt,
        )

        _set_stage(5, _STAGE_LABELS[4])
        stage_bar.progress(1.0, text="Complete!")
        candidate_label.markdown(
            f"✅ {result.succeeded} stencil pack(s) exported, "
            f"{result.failed} error(s)."
        )
        log_placeholder.empty()

        st.session_state.last_result = result
        st.session_state.last_output_dir = output_dir
        st.rerun()

    except Exception as exc:  # noqa: BLE001
        stage_bar.progress(0, text="Failed")
        st.error(f"Generation failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()
