from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from gruut import sentences
from g2p_id import G2p
from nltk.tokenize import sent_tokenize, TweetTokenizer
from functools import lru_cache
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import PreTrainedTokenizerFast

from . import gruut_symbols
from . import gruut_sw_symbols
from . import g2p_id_symbols
from .normalization import preprocess_text

import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], symbols: List[str]):
        if emphasis_model_path:
            self.model, self.tokenizer = self.load_model_and_tokenizer(emphasis_model_path)
        if emphasis_lookup:
            self.emphasis_lookup = emphasis_lookup
        self.escaped_symbols = self.prepare_escaped_symbols(symbols)
        
    @abstractmethod
    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[List[str], str]:
        pass

    @abstractmethod
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        pass
    
    @abstractmethod
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        pass
            
    @staticmethod
    def load_model_and_tokenizer(model_dir):
        model = ORTModelForQuestionAnswering.from_pretrained(model_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        return model, tokenizer

    def infer(self, input_phonemes: str) -> tuple:
        splitted_phonemes = self.split_phonemes(input_phonemes)
        inputs = self.tokenizer(" ".join(splitted_phonemes), return_tensors="pt")
        outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        
        start_index = np.argmax(outputs.start_logits)
        end_index = np.argmax(outputs.end_logits)
            
        return start_index.item(), end_index.item(), splitted_phonemes

    @lru_cache(maxsize=1000)
    def emphasize_phonemes(self, phonemes: str) -> str:
        start_idx, end_idx, splitted_phonemes = self.infer(phonemes)
        emphasized = self.postprocess_prediction(splitted_phonemes, start_idx, end_idx)
        return emphasized

    def split_phonemes(self, input_string: str) -> List[str]:
        # remove spaces before end of punctuations
        input_string = re.sub(r'\s+([,.;?!])', r'\1', input_string)
        return re.findall(self.escaped_symbols, input_string)

    @staticmethod
    def postprocess_prediction(phonemes: str, start_idx: int, end_idx: int) -> str:
        return ''.join(phonemes[:start_idx] + ['"'] + phonemes[start_idx:end_idx+1] + ['"'] + phonemes[end_idx+1:])

    def prepare_escaped_symbols(self, symbols: List[str]):
        escaped_symbols = [re.escape(symbol) for symbol in symbols]
        escaped_symbols.sort(key=lambda x: -len(x))
        return '|'.join(escaped_symbols)
    

class GruutTokenizer(BaseTokenizer):
    
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str]):
        super().__init__(emphasis_model_path, emphasis_lookup, gruut_symbols.SYMBOLS)
    
    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[List[str], str]:
        text = preprocess_text(text, normalize)
        phonemes = []
        words = []
        in_quotes = False
        
        for sentence in sentences(text, lang="en"):
            for word in sentence:
                if word.text == '"':
                    phonemes, words, in_quotes = self.handle_quote(phonemes, words, in_quotes)
                elif word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                elif word.phonemes:
                    phonemes, words = self.handle_word(phonemes, words, word, in_quotes)
        
        return ''.join(phonemes), text

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        return gruut_symbols.phonemes_to_ids(phonemes)
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        return gruut_symbols.ids_to_phonemes(ids)
    
    def handle_quote(self, phonemes, words, in_quotes):
        if in_quotes and words:
            phonemes, words = self.handle_emphasized_word(phonemes, words)
            in_quotes = False
        else:
            if phonemes and phonemes[-1] != ' ':
                phonemes.append(' ')
            phonemes.append('"')
            in_quotes = True
        return phonemes, words, in_quotes

    def handle_emphasized_word(self, phonemes, words):
        try:
            emphasized_phonemes = self.emphasis_lookup[words[-1]]
        except KeyError:
            emphasized_phonemes = self.emphasize_phonemes(phonemes[-1])
        phonemes = phonemes[:-2] + [emphasized_phonemes]
        words.pop()
        return phonemes, words

    def handle_word(self, phonemes, words, word, in_quotes):
        if not in_quotes and phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        phonemes.append(''.join(word.phonemes))
        if in_quotes:
            words.append(word.text)
        return phonemes, words


class GruutSwahiliTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str]):
        super().__init__(emphasis_model_path, emphasis_lookup, gruut_sw_symbols.SYMBOLS)
        
    def phonemize_text(self, text: str, normalize: bool = False) -> str:
        text = preprocess_text(text, normalize)
        phonemes = []
        for sentence in sentences(text, lang="sw"):
            sent_ph = []
            for idx, word in enumerate(sentence):
                if word.is_major_break or word.is_minor_break:
                    sent_ph.append(word.text)
                elif word.text == '"':
                    sent_ph.append('"')
                elif word.phonemes:
                    sent_ph += word.phonemes

                if word.trailing_ws:
                    sent_ph.append(" ")
            phonemes += sent_ph
        return ''.join(phonemes), text

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        return gruut_sw_symbols.phonemes_to_ids(phonemes)

    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        return gruut_sw_symbols.ids_to_phonemes(ids)

class G2pIdTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str]):
        super().__init__(emphasis_model_path, emphasis_lookup, g2p_id_symbols.SYMBOLS)
        self.g2p = G2p()
        self.tokenizer = TweetTokenizer()
        self.puncts = ".,!?:"

    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[List[str], str]:
        text = preprocess_text(text, normalize)
        phonemes = []
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            start_quote = False
            words = self.tokenizer.tokenize(sentence)
            sent_ph = self.g2p(sentence)

            for idx, word in enumerate(words):
                if word == '"':
                    sent_ph.insert(idx, '"')
            assert len(words) == len(sent_ph)

            for idx, word in enumerate(sent_ph):
                phonemes += word
                if word == '"':
                    if start_quote:
                        start_quote = False
                    else:
                        start_quote = True
                        continue

                if idx < len(sent_ph) - 1 and all(p not in self.puncts for p in sent_ph[idx + 1]) and not start_quote:
                    phonemes += [" "]
                    
            # Add space after the sentence if it's not the last sentence
            if i < len(sentences) - 1:
                phonemes += [" "]

        return ''.join(phonemes), text

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        return g2p_id_symbols.phonemes_to_ids(phonemes)
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        return g2p_id_symbols.ids_to_phonemes(ids)

class Tokenizer:
    def __init__(self, model_dirs: Dict[str, str], emphasis_lookup: Dict[str, str]):
        self.tokenizers = {
            "en": GruutTokenizer(model_dirs["en"], emphasis_lookup),
            "sw": GruutSwahiliTokenizer(model_dirs["sw"], emphasis_lookup),
            "id": G2pIdTokenizer(model_dirs["id"], emphasis_lookup),
        }

    def get_tokenizer(self, language: str) -> BaseTokenizer:
        try:
            return self.tokenizers[language]
        except KeyError:
            raise ValueError(f"Unsupported language: {language}")
