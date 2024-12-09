import re
from typing import List


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

def prepare_escaped_symbols(symbols: List[str]):
    # Sort symbols by length (longest first) to ensure correct matching
    sorted_symbols = sorted(symbols, key=len, reverse=True)

    escaped_patterns = [re.escape(s) for s in sorted_symbols]

    # Join all patterns with | for alternation
    return '|'.join(escaped_patterns)


def split_phonemes(input_string: str, symbols: List[str]) -> List[str]:
    escaped_symbols = prepare_escaped_symbols(symbols)
    escaped_symbols_pattern = re.compile(escaped_symbols)
    input_string = input_string.replace(".", "")
    word_phonemes = input_string.split(" ")
    print(f"word_phonemes: {word_phonemes}")
    
    result = []
    for i, word_ipa in enumerate(word_phonemes):
        result.extend(escaped_symbols_pattern.findall(word_ipa))
        if i < len(word_phonemes) - 1:
            result.append(' ')
    
    return " ".join(result)

if __name__ == "__main__":
    print(split_phonemes("ə.bˈæn.dən", ENGLISH_SYMBOLS))