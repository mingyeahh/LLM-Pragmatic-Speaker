"""Building the utterance space: logical alternatives and top-k beam search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd

from rsa_speaker.beam_search import beam_search
from rsa_speaker.corpus import Furniture, parse_furniture, produce_prompt

if TYPE_CHECKING:
    from llama_cpp import Llama


_ORIENTATION_PHRASES = {
    "left": " facing to the left",
    "right": " facing to the right",
    "front": " facing forwards",
    "back": " facing backwards",
    None: "",
}


@dataclass
class FurnitureDescription:
    """A furniture description where any feature may be omitted (None)."""

    size: str | None
    colour: str | None
    furn_type: str | None
    orientation: str | None

    def __str__(self) -> str:
        noun = self.furn_type if self.furn_type is not None else "thing"
        adjectives = [f for f in (self.size, self.colour) if f is not None]
        orientation = _ORIENTATION_PHRASES[self.orientation]
        if not adjectives:
            return f"a {noun}" + orientation
        if len(adjectives) == 1:
            return f"a {adjectives[0]} {noun}" + orientation
        return f"a {self.size}, {self.colour} {noun}" + orientation

    def describes(self, furniture: Furniture) -> bool:
        if self.size is not None and furniture.size != self.size:
            return False
        if self.colour is not None and furniture.colour != self.colour:
            return False
        if self.furn_type is not None and furniture.furn_type != self.furn_type:
            return False
        if self.orientation is not None and furniture.orientation != self.orientation:
            return False
        return True


def furniture_feature_space(objects: list[Furniture]) -> dict[str, set[str | None]]:
    space: dict[str, set[str | None]] = {
        "size": set(),
        "colour": set(),
        "furn_type": set(),
        "orientation": set(),
    }
    for obj in objects:
        space["size"].add(obj.size)
        space["colour"].add(obj.colour)
        space["furn_type"].add(obj.furn_type)
        space["orientation"].add(obj.orientation)
    for key in space:
        space[key].add(None)
    return space


def logical_alternatives(objects: list[Furniture]) -> Iterator[str]:
    """Every constructed utterance that describes at least one object in the room."""
    space = furniture_feature_space(objects)
    for combo in product(
        space["size"], space["colour"], space["furn_type"], space["orientation"]
    ):
        desc = FurnitureDescription(*combo)
        if any(desc.describes(obj) for obj in objects):
            yield str(desc)


def write_logical_file(corpus_file: str | Path, out_file: str | Path) -> None:
    # first line is the source corpus path, the rest are utterances
    objects = parse_furniture(corpus_file)
    with open(out_file, "w") as f:
        f.write(f"{corpus_file}\n")
        for utt in logical_alternatives(objects):
            f.write(utt + "\n")


def read_logical_file(txt_file: str | Path) -> tuple[str, list[str]]:
    lines = Path(txt_file).read_text().splitlines()
    return lines[0].strip(), [line.rstrip() for line in lines[1:]]


def generate_topk(
    corpus_file: str,
    model: "Llama",
    beam_width: int,
    n_objects: int = 7,
) -> pd.DataFrame:
    """Beam-search top-k utterances for every object in a furniture trial."""
    objects = parse_furniture(corpus_file)
    frames = []
    for target in range(n_objects):
        prompt = produce_prompt(objects, target) + "\n"
        beams = beam_search(prompt, model, beam_width)
        frames.append(
            pd.DataFrame(
                [[utt, target, logp] for utt, logp in beams.items()],
                columns=["Sequence", "Object", "p_llm"],
            )
        )
    return pd.concat(frames, ignore_index=True)
