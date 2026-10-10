"""Local Gradio app for packaging-type review (PLAN 1.7) and concept labelling (PLAN 1.8).

The app never shows an image's authentic/counterfeit label, so annotations stay label-blind.
It binds to 127.0.0.1 only (no public share link) and saves after every click.

Usage:
    python -m pharmachecked.data.annotation_app --mode packaging
    python -m pharmachecked.data.annotation_app --mode concepts
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotations import (
    CONCEPT_VALUES,
    CONCEPTS,
    PACKAGING_TYPES,
    ConceptStore,
    PackagingStore,
)

ANNOTATIONS = REPO_ROOT / "data" / "annotations"
IMAGES = REPO_ROOT / "data" / "processed" / "images"


def step(
    ids: list[str], index: int, delta: int, is_done: Callable[[str], bool], todo_only: bool
) -> int:
    """Index of the next (delta=+1) or previous (delta=-1) image, optionally skipping done ones.

    Stays at ``index`` when there is nothing further in that direction.
    """
    k = index + delta
    while 0 <= k < len(ids):
        if not todo_only or not is_done(ids[k]):
            return k
        k += delta
    return index


def first_todo(ids: list[str], is_done: Callable[[str], bool]) -> int:
    """Index of the first image without an annotation (0 if all are done)."""
    return next((k for k, i in enumerate(ids) if not is_done(i)), 0)


def _image(image_id: str, images_dir: Path) -> Image.Image:
    with Image.open(images_dir / f"{image_id}.png") as im:
        return im.convert("RGB")


def build_packaging_app(
    store: PackagingStore, images_dir: Path = IMAGES, only_label: str | None = None
) -> gr.Blocks:
    """Packaging-type review: CLIP suggestion pre-selected, one click to confirm or correct.

    With ``only_label``, shows just the images currently labelled that type (re-review), and an
    image counts as done once it has been saved in this session.
    """
    if only_label is None:
        ids = store.ids()
    else:
        ids = [i for i in store.ids() if store.row(i)["final_label"] == only_label]
        if not ids:
            raise ValueError(f"no images are labelled {only_label!r}")
    reviewed: set[str] = set()

    def done(image_id: str) -> bool:
        if only_label is not None:
            return image_id in reviewed
        return store.row(image_id)["final_label"] != ""

    def view(index: int) -> tuple[Any, ...]:
        row = store.row(ids[index])
        status = f"labelled **{row['final_label']}**" if row["final_label"] else "not labelled yet"
        info = (
            f"Image {index + 1} / {len(ids)} · {store.n_done()} done · `{ids[index]}`  \n"
            f"CLIP suggests **{row['clip_label']}** ({float(row['clip_conf']):.2f}) · {status}"
        )
        return index, _image(ids[index], images_dir), info, row["final_label"] or row["clip_label"]

    def save(index: int, label: str, annotator: str, todo_only: bool) -> tuple[Any, ...]:
        if not annotator.strip():
            raise gr.Error("Enter your name in 'Annotator' first.")
        store.set_label(ids[index], label, annotator)
        reviewed.add(ids[index])
        return view(step(ids, index, +1, done, todo_only))

    with gr.Blocks(title="Packaging type review") as app:
        scope = (
            f"Re-reviewing the {len(ids)} images currently labelled **{only_label}**."
            if only_label
            else "Choose the packaging type shown in the photo."
        )
        gr.Markdown(f"## Packaging type review\n{scope}")
        index = gr.State(first_todo(ids, done))
        with gr.Row():
            annotator = gr.Textbox(label="Annotator", scale=2)
            todo_only = gr.Checkbox(
                value=True,
                label="Skip images already saved this session"
                if only_label
                else "Skip images already labelled",
            )
        info = gr.Markdown()
        image = gr.Image(type="pil", format="png", height=520, interactive=False, show_label=False)
        label = gr.Radio(list(PACKAGING_TYPES), label="Packaging type")
        with gr.Row():
            prev_btn = gr.Button("◀ Previous")
            save_btn = gr.Button("Save and next ▶", variant="primary")
            next_btn = gr.Button("Skip ▶")
        outputs = [index, image, info, label]
        app.load(view, inputs=index, outputs=outputs)
        save_btn.click(save, inputs=[index, label, annotator, todo_only], outputs=outputs)
        prev_btn.click(lambda k: view(step(ids, k, -1, done, False)), inputs=index, outputs=outputs)
        next_btn.click(
            lambda k, t: view(step(ids, k, +1, done, t)), inputs=[index, todo_only], outputs=outputs
        )
    return app


def build_concept_app(store: ConceptStore, images_dir: Path = IMAGES) -> gr.Blocks:
    """Concept labelling: seven present/absent/unsure choices per image."""
    ids = store.ids()

    def done(image_id: str) -> bool:
        row = store.row(image_id)
        return all(row[c] != "" for c in CONCEPTS)

    def view(index: int) -> tuple[Any, ...]:
        row = store.row(ids[index])
        info = (
            f"Image {index + 1} / {len(ids)} · {store.n_done()} done · `{ids[index]}` · "
            f"{'saved' if done(ids[index]) else 'not saved yet'}"
        )
        values = [row[c] or None for c in CONCEPTS]
        return (index, _image(ids[index], images_dir), info, *values)

    def save(index: int, annotator: str, todo_only: bool, *values: str | None) -> tuple[Any, ...]:
        if not annotator.strip():
            raise gr.Error("Enter your name in 'Annotator' first.")
        if any(v is None for v in values):
            raise gr.Error("Choose present, absent or unsure for every concept.")
        store.set_concepts(ids[index], dict(zip(CONCEPTS, values, strict=True)), annotator)
        return view(step(ids, index, +1, done, todo_only))

    with gr.Blocks(title="Concept labelling") as app:
        gr.Markdown("## Concept labelling\nFor each concept, is it visible in this photo?")
        index = gr.State(first_todo(ids, done))
        with gr.Row():
            annotator = gr.Textbox(label="Annotator", scale=2)
            todo_only = gr.Checkbox(value=True, label="Skip images already labelled")
        info = gr.Markdown()
        with gr.Row():
            image = gr.Image(type="pil", format="png", height=560, interactive=False, scale=3)
            with gr.Column(scale=2):
                radios = [
                    gr.Radio(list(CONCEPT_VALUES), label=c.replace("_", " ")) for c in CONCEPTS
                ]
        with gr.Row():
            prev_btn = gr.Button("◀ Previous")
            save_btn = gr.Button("Save and next ▶", variant="primary")
            next_btn = gr.Button("Skip ▶")
        outputs = [index, image, info, *radios]
        app.load(view, inputs=index, outputs=outputs)
        save_btn.click(save, inputs=[index, annotator, todo_only, *radios], outputs=outputs)
        prev_btn.click(lambda k: view(step(ids, k, -1, done, False)), inputs=index, outputs=outputs)
        next_btn.click(
            lambda k, t: view(step(ids, k, +1, done, t)), inputs=[index, todo_only], outputs=outputs
        )
    return app


def main(argv: list[str] | None = None) -> int:
    """Launch the app locally; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("packaging", "concepts"), required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--review",
        choices=PACKAGING_TYPES,
        default=None,
        help="packaging mode: re-review only the images currently labelled this type",
    )
    args = parser.parse_args(argv)
    if args.mode == "packaging":
        path = ANNOTATIONS / "packaging_type.csv"
        if not path.exists():
            sys.stderr.write("run scripts/prelabel_packaging.py first\n")
            return 1
        app = build_packaging_app(PackagingStore(path), only_label=args.review)
    else:
        path = ANNOTATIONS / "concepts.csv"
        if not path.exists():
            sys.stderr.write("run scripts/select_concept_subset.py first\n")
            return 1
        app = build_concept_app(ConceptStore(path))
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
