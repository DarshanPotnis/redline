"""Logic-grid vocabulary: six neutral attributes, six values each, and one phrase per attribute."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute:
    name: str
    label: str
    values: tuple[str, ...]
    template: str  # how a clue names the person holding a value; "{}" is the value

    def phrase(self, value: str) -> str:
        return self.template.format(value)


# Name is always first and always used. Values carry no world-knowledge links (no cat/milk or
# fish/water pairs), and none is a substring of another, so answers parse unambiguously. No
# template, label or value uses a position word ("house", "left", "right", "next", a number):
# those belong to the position clues alone. Text v1 said "the person in the blue house", which
# models read as a house position (pilot-1e); v2 says favorite color.
VOCAB: tuple[Attribute, ...] = (
    Attribute("name", "Name", ("Alice", "Bob", "Carol", "Dave", "Erin", "Frank"), "{}"),
    Attribute(
        "color",
        "Favorite color",
        ("red", "green", "blue", "yellow", "white", "purple"),
        "the person whose favorite color is {}",
    ),
    Attribute(
        "pet",
        "Pet",
        ("cat", "dog", "bird", "fish", "rabbit", "hamster"),
        "the person who keeps the {}",
    ),
    Attribute(
        "drink",
        "Drink",
        ("tea", "coffee", "juice", "soda", "lemonade", "cocoa"),
        "the person who drinks {}",
    ),
    Attribute(
        "hobby",
        "Hobby",
        ("chess", "hiking", "painting", "pottery", "knitting", "cycling"),
        "the person who enjoys {}",
    ),
    Attribute(
        "music",
        "Music",
        ("jazz", "rock", "pop", "folk", "techno", "reggae"),
        "the person who likes {} music",
    ),
)

ATTRIBUTES: dict[str, Attribute] = {attribute.name: attribute for attribute in VOCAB}
