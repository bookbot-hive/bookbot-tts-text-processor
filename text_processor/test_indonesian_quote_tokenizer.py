"""
Regression test for Indonesian smart-quote punctuation handling.

Issue this covers:
    The production log showed an IndexError in G2pIdTokenizer.phonemize_text()
    for this input:

        “Aku harus berubah,” tekad Tia dalam hati.

    Before the fix, the Indonesian tokenizer split words only on a small set of
    trailing punctuation. Curly quotes stayed attached to words like "“Aku" and
    "berubah,”". g2p_id could then return fewer phoneme entries than the token
    loop expected, so tokenizers.py attempted sent_ph[ph_idx] past the end of
    the list.

What this test verifies:
    - Smart quotes are normalized/stripped before Indonesian G2P word handling.
    - Comma and period remain punctuation in the output.
    - The old sent_ph[ph_idx] IndexError path is not reached.

This file stubs heavy optional dependencies and fake-G2Ps each word into its
lowercase letters. It is a tokenizer regression test, not a pronunciation
quality test.
"""

import importlib
import sys
import types
from pathlib import Path


FAILING_TEXT = "“Aku harus berubah,” tekad Tia dalam hati."
EXPECTED_PHONEMES = "aku harus berubah, tekad tia dalam hati."
EXPECTED_WORD_BOUNDARIES = 7


def _install_dependency_stubs():
    nltk = types.ModuleType("nltk")
    nltk.download = lambda *args, **kwargs: True
    nltk_tokenize = types.ModuleType("nltk.tokenize")
    nltk_tokenize.sent_tokenize = lambda text: [text]
    nltk.tokenize = nltk_tokenize
    sys.modules["nltk"] = nltk
    sys.modules["nltk.tokenize"] = nltk_tokenize

    gruut = types.ModuleType("gruut")
    gruut.sentences = lambda *args, **kwargs: []
    sys.modules["gruut"] = gruut

    g2p_id = types.ModuleType("g2p_id")

    class FakeG2p:
        def __init__(self, turso_config=None):
            pass

        def __call__(self, text):
            return [[char for char in text.lower() if char.isalpha()]]

    g2p_id.G2p = FakeG2p
    sys.modules["g2p_id"] = g2p_id

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda *args, **kwargs: ""
    sys.modules["huggingface_hub"] = huggingface_hub

    huggingface_errors = types.ModuleType("huggingface_hub.errors")

    class EntryNotFoundError(Exception):
        pass

    huggingface_errors.EntryNotFoundError = EntryNotFoundError
    sys.modules["huggingface_hub.errors"] = huggingface_errors

    transformers = types.ModuleType("transformers")

    class PreTrainedTokenizerFast:
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls()

    transformers.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    sys.modules["transformers"] = transformers

    onnxruntime = types.ModuleType("onnxruntime")
    onnxruntime.InferenceSession = lambda *args, **kwargs: object()
    sys.modules["onnxruntime"] = onnxruntime

    numpy = types.ModuleType("numpy")
    sys.modules.setdefault("numpy", numpy)


def _ensure_local_package():
    package_dir = Path(__file__).resolve().parent
    package_name = package_dir.name

    if str(package_dir.parent) not in sys.path:
        sys.path.insert(0, str(package_dir.parent))

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    return package_name


def _install_package_stubs(package_name):
    utils = types.ModuleType(f"{package_name}.utils")

    class TextUtils:
        UNICODE_NORM_FORM = "NFKC"

        @staticmethod
        def get_custom_tags():
            return {}

    utils.TextUtils = TextUtils
    sys.modules[f"{package_name}.utils"] = utils


def main():
    _install_dependency_stubs()
    package_name = _ensure_local_package()
    _install_package_stubs(package_name)
    tokenizers = importlib.import_module(f"{package_name}.tokenizers")

    tokenizer = tokenizers.G2pIdTokenizer(
        emphasis_model_path=None,
        emphasis_lookup={},
        language="id",
    )
    phonemes, normalized_text, word_boundaries = tokenizer.phonemize_text(
        FAILING_TEXT
    )

    print(f"Input: {FAILING_TEXT}")
    print(f"Phonemes: {phonemes}")
    print(f"Word boundaries: {word_boundaries}")

    assert "Aku" in normalized_text
    assert phonemes == EXPECTED_PHONEMES
    assert len(word_boundaries) == EXPECTED_WORD_BOUNDARIES
    assert "“" not in phonemes
    assert "”" not in phonemes

    print("OK: Indonesian smart-quote tokenizer regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
