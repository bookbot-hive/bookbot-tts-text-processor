from .utils import TextUtils
import re
import unicodedata


def preprocess_text(text: str, normalize: bool = False) -> str:
    if normalize:
        text = unicodedata.normalize(TextUtils.UNICODE_NORM_FORM, text)
    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # remove spaces before punctuation
    text = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", text)

    # ensure space after comma, colon, semicolon (but not if followed by whitespace or end of string)
    text = re.sub(r"([,;:])(?!\s|$)", r"\1 ", text)
    return text.strip()


if __name__ == "__main__":
    print(preprocess_text("Hello,world!This is a test."))
