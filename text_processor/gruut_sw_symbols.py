import re
from .utils import CUSTOM_TAGS

SYMBOLS = [
    "_",
    "^",
    "$",
    " ",
    "!",
    '"',
    "#",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    "?",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "s",
    "t",
    "t͡ʃ",
    "u",
    "v",
    "w",
    "x",
    "z",
    "ð",
    "ŋ",
    "ɑ",
    "ɓ",
    "ɔ",
    "ɗ",
    "ɛ",
    "ɠ",
    "ɣ",
    "ɾ",
    "ʃ",
    "ʄ",
    "θ",
    "ᵐɓ",
    "ᵑg",
    "ᶬv",
    "ⁿz",
    "ⁿɗ",
    "ⁿɗ͡ʒ",
]


# Special symbols
PAD = "_"
BOS = "^"
EOS = "$"

# Special symbol ids
PAD_ID = SYMBOLS.index(PAD)
BOS_ID = SYMBOLS.index(BOS)
EOS_ID = SYMBOLS.index(EOS)
SPACE_ID = SYMBOLS.index(" ")

# Mappings from symbol to numeric ID and vice versa:
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}  # pylint: disable=unnecessary-comprehension


def phonemes_to_ids(phonemes):
    """Converts a list of phonemes and tags to a sequence of IDs."""
    sequence = []
    tag_pattern = re.compile(r'<(\w+)>')
    
    for phoneme in phonemes:
        # Check if it's a tag
        match = tag_pattern.match(phoneme)
        if match:
            tag = match.group(1)  # Get the tag name without <>
            if tag in CUSTOM_TAGS:
                sequence.append(CUSTOM_TAGS[tag])
            else:
                raise ValueError(f"Unknown tag: {tag}")
        else:
            # Regular phoneme
            if phoneme in SYMBOL_TO_ID:
                sequence.append(SYMBOL_TO_ID[phoneme])
            else:
                raise ValueError(f"Unknown phoneme: {phoneme}")
    
    return sequence

def ids_to_phonemes(sequence):
    """Converts a sequence of IDs back to a string, including special tags"""
    result = ""
    for symbol_id in sequence:
        # Handle special tag IDs (negative numbers)
        if symbol_id < 0:
            # Find the tag name by value in CUSTOM_TAGS
            tag_name = next(tag for tag, id in CUSTOM_TAGS.items() if id == symbol_id)
            result += f"<{tag_name}>"
        else:
            # Regular phoneme ID
            s = ID_TO_SYMBOL[symbol_id]
            result += s
    return result