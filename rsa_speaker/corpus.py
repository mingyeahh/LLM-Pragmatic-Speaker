"""Parsing the TUNA corpus and building reference-game prompts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ORIENTATION_PHRASES = {
    "left": "to the left",
    "right": "to the right",
    "front": "forwards",
    "back": "backwards",
}


@dataclass
class Furniture:
    size: str
    colour: str
    furn_type: str
    orientation: str

    def __str__(self) -> str:
        return (
            f"a {self.size}, {self.colour} {self.furn_type} facing "
            + ORIENTATION_PHRASES[self.orientation]
        )


@dataclass
class People:
    age: str
    orientation: str
    hair_colour: str
    has_suit: bool
    has_shirt: bool
    has_tie: bool
    has_beard: bool
    has_glasses: bool
    has_hair: bool

    def __str__(self) -> str:
        age_map = {"old": "An old person", "young": "A young person"}
        out = age_map[self.age]
        out += " facing " + ORIENTATION_PHRASES[self.orientation]

        if self.has_hair:
            out += f" that has {self.hair_colour} hair"
        else:
            out += " that is bald"

        clothes = []
        if self.has_shirt:
            clothes.append("a shirt")
        if self.has_suit:
            clothes.append("a suit")
        if len(clothes) == 1:
            out += ", wears " + clothes[0]
        elif len(clothes) == 2:
            out += ", wears " + clothes[0] + " and " + clothes[1]

        extras = []
        if self.has_tie:
            extras.append("a tie")
        if self.has_glasses:
            extras.append("glasses")
        if len(extras) == 1:
            out += ", and has " + extras[0]
        elif len(extras) == 2:
            out += ", and has " + extras[0] + " and " + extras[1]
        return out + "."


def _read_entities(filename: str | Path) -> list[dict[str, str]]:
    dom = ET.parse(filename).find("DOMAIN")
    if dom is None:
        raise ValueError(f"No <DOMAIN> element found in {filename}")
    entities = []
    for entity in dom.findall("ENTITY"):
        attrs = {a.get("NAME"): a.get("VALUE") for a in entity.findall("ATTRIBUTE")}
        entities.append(attrs)
    return entities


def parse_furniture(filename: str | Path) -> list[Furniture]:
    return [
        Furniture(
            size=e["size"],
            colour=e["colour"],
            furn_type=e["type"],
            orientation=e["orientation"],
        )
        for e in _read_entities(filename)
    ]


def parse_furniture_features(filename: str | Path) -> list[dict[str, str]]:
    """Object attribute dicts, dropping the gradable x/y dimensions."""
    skip = {"x-dimension", "y-dimension"}
    return [{k: v for k, v in e.items() if k not in skip} for e in _read_entities(filename)]


def parse_people(filename: str | Path) -> list[People]:
    out = []
    for e in _read_entities(filename):
        out.append(
            People(
                age=e["age"].strip(),
                orientation=e["orientation"].strip(),
                hair_colour=e["hairColour"].strip(),
                has_suit=e["hasSuit"] == "1",
                has_shirt=e["hasShirt"] == "1",
                has_tie=e["hasTie"] == "1",
                has_beard=e["hasBeard"] == "1",
                has_glasses=e["hasGlasses"] == "1",
                has_hair=e["hasHair"] == "1",
            )
        )
    return out


def produce_prompt(domain: list[Furniture | People], target_index: int) -> str:
    if not domain:
        raise ValueError("domain must contain at least one object")

    is_people = isinstance(domain[0], People)
    plural = "people" if is_people else "objects"
    singular = "person" if is_people else "object"

    prompt = f"There are {len(domain)} {plural} in a room:\n\n"
    for i, obj in enumerate(domain):
        prompt += f"{i + 1}. {obj}\n"
    prompt += (
        f"\nIdentify {singular} {target_index + 1} in the room to distinguish from "
        "other objects using the fewest possible words (not numbers).\n\nAnswer: "
    )
    return prompt
