from .utils import UNICODE_NORM_FORM
import re
import unicodedata

def preprocess_text(text: str, normalize: bool = False) -> str:
    if normalize:
        text = unicodedata.normalize(UNICODE_NORM_FORM, text)
    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # remove spaces before punctuation
    text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
    return text.strip()