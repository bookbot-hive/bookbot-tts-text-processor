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
    "aɪ",
    "aʊ",
    "b",
    "d",
    "d͡ʒ",
    "eɪ",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "oʊ",
    "p",
    "s",
    "t",
    "t͡ʃ",
    "u",
    "v",
    "w",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɑ",
    "ɔ",
    "ɔɪ",
    "ə",
    "ɚ",
    "ɛ",
    "ɡ",
    "ɪ",
    "ɹ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʒ",
    "ˈaɪ",
    "ˈaʊ",
    "ˈeɪ",
    "ˈi",
    "ˈoʊ",
    "ˈu",
    "ˈæ",
    "ˈɑ",
    "ˈɔ",
    "ˈɔɪ",
    "ˈɚ",
    "ˈɛ",
    "ˈɪ",
    "ˈʊ",
    "ˈʌ",
    "ˌaɪ",
    "ˌaʊ",
    "ˌeɪ",
    "ˌi",
    "ˌoʊ",
    "ˌu",
    "ˌæ",
    "ˌɑ",
    "ˌɔ",
    "ˌɔɪ",
    "ˌɚ",
    "ˌɛ",
    "ˌɪ",
    "ˌʊ",
    "ˌʌ",
    "θ",
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
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = ID_TO_SYMBOL[symbol_id]
        result += s
    return result
