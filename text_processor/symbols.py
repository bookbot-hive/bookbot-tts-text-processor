import re
from .utils import TextUtils

class SymbolSet:
    def __init__(self, symbols):
        self.SYMBOLS = symbols
        
        # Special symbols
        self.PAD = "_"
        self.BOS = "^"
        self.EOS = "$"
        
        # Special symbol ids
        self.PAD_ID = self.SYMBOLS.index(self.PAD)
        self.BOS_ID = self.SYMBOLS.index(self.BOS)
        self.EOS_ID = self.SYMBOLS.index(self.EOS)
        self.SPACE_ID = self.SYMBOLS.index(" ")
        
        # Mappings
        self.SYMBOL_TO_ID = {s: i for i, s in enumerate(self.SYMBOLS)}
        self.ID_TO_SYMBOL = {i: s for i, s in enumerate(self.SYMBOLS)}

    def phonemes_to_ids(self, phonemes):
        """Converts a list of phonemes and tags to a sequence of IDs."""
        sequence = []
        tag_pattern = re.compile(r'<([^>]+)>')
        
        for phoneme in phonemes:
            match = tag_pattern.match(phoneme)
            print(f"Match: {match}")
            if match:
                tag = match.group(1)
                custom_tags = TextUtils.get_custom_tags()
                if tag in custom_tags:
                    sequence.append(custom_tags[tag])
                else:
                    raise ValueError(f"Unknown tag: {tag}")
            else:
                if phoneme in self.SYMBOL_TO_ID:
                    sequence.append(self.SYMBOL_TO_ID[phoneme])
                else:
                    raise ValueError(f"Unknown phoneme: {phoneme}")
        
        return sequence

    def ids_to_phonemes(self, sequence):
        """Converts a sequence of IDs back to a string, including special tags"""
        result = ""
        custom_tags = TextUtils.get_custom_tags()
        
        for symbol_id in sequence:
            if symbol_id < 0:
                tag_name = next(tag for tag, id in custom_tags.items() if id == symbol_id)
                result += f"<{tag_name}>"
            else:
                s = self.ID_TO_SYMBOL[symbol_id]
                result += s
        return result


# Define language-specific symbol sets
ENGLISH_SYMBOLS = [
    "_", "^", "$", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".", ":", ";", "?",
    "aɪ", "aʊ", "b", "d", "d͡ʒ", "eɪ", "f", "h", "i", "j", "k", "l", "m", "n", "oʊ",
    "p", "s", "t", "t͡ʃ", "u", "v", "w", "z", "æ", "ð", "ŋ", "ɑ", "ɔ", "ɔɪ", "ə",
    "ɚ", "ɛ", "ɡ", "ɪ", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "ˈaɪ", "ˈaʊ", "ˈeɪ", "ˈi", "ˈoʊ",
    "ˈu", "ˈæ", "ˈɑ", "ˈɔ", "ˈɔɪ", "ˈɚ", "ˈɛ", "ˈɪ", "ˈʊ", "ˈʌ", "ˌaɪ", "ˌaʊ",
    "ˌeɪ", "ˌi", "ˌoʊ", "ˌu", "ˌæ", "ˌɑ", "ˌɔ", "ˌɔɪ", "ˌɚ", "ˌɛ", "ˌɪ", "ˌʊ",
    "ˌʌ", "θ",
]

INDONESIAN_SYMBOLS = [
    "_", "^", "$", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".", ":", ";", "?",
    "a", "b", "tʃ", "d", "e", "f", "ɡ", "h", "i", "dʒ", "k", "l", "m", "n", "o",
    "p", "r", "s", "t", "u", "v", "w", "j", "z", "ŋ", "ə", "ɲ", "ʃ", "x", "ʔ",
]

SWAHILI_SYMBOLS = [
    "_", "^", "$", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".", ":", ";", "?",
    "f", "h", "i", "j", "k", "l", "m", "n", "p", "s", "t", "t͡ʃ", "u", "v", "w",
    "x", "z", "ð", "ŋ", "ɑ", "ɓ", "ɔ", "ɗ", "ɛ", "ɠ", "ɣ", "ɾ", "ʃ", "ʄ", "θ",
    "ᵐɓ", "ᵑg", "ᶬv", "ⁿz", "ⁿɗ", "ⁿɗ͡ʒ",
]

# Create language-specific symbol sets
english_symbols = SymbolSet(ENGLISH_SYMBOLS)
indonesian_symbols = SymbolSet(INDONESIAN_SYMBOLS)
swahili_symbols = SymbolSet(SWAHILI_SYMBOLS)

# Factory function to get the appropriate symbol set
def get_symbol_set(language: str) -> SymbolSet:
    """Get the appropriate symbol set for a given language code."""
    symbol_sets = {
        "en": english_symbols,
        "id": indonesian_symbols,
        "sw": swahili_symbols
    }
    if language not in symbol_sets:
        raise ValueError(f"Unsupported language code: {language}")
    return symbol_sets[language]