import torch
import logging
import re
import json
from typing import List
from gruut import sentences
from transformers import PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForQuestionAnswering
from functools import lru_cache
from .utils import IPA_LIST
from .gruut_symbols import phonemes_to_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmphasisModel:
    def __init__(self, model_dir: str, db_path: str):
        # ... existing code ...
        self.device = torch.device("cpu")
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_dir)
        self.load_emphasis_lookup(db_path)
        self.escaped_symbols = self.prepare_escaped_symbols()
        
    def load_emphasis_lookup(self, db_path: str):
        with open(db_path, 'r') as f:
            self.emphasis_lookup = json.load(f)

    @staticmethod
    def load_model_and_tokenizer(model_dir):
        model = ORTModelForQuestionAnswering.from_pretrained(model_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        
        return model, tokenizer

    def infer(self, input_phonemes: str) -> tuple:
        with torch.no_grad():
            preprocessed_phonemes = self.preprocess_phonemes(input_phonemes)
            inputs = self.tokenizer(" ".join(preprocessed_phonemes), return_tensors="pt")
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            start_index = torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits)
            
        return start_index.item(), end_index.item(), preprocessed_phonemes
    
    @lru_cache(maxsize=1000)
    def emphasize_phonemes(self, phonemes: str) -> str:
        start_idx, end_idx, preprocessed_phonemes = self.infer(phonemes)
        emphasized = self.postprocess_prediction(preprocessed_phonemes, start_idx, end_idx)
        return emphasized
    
    def get_input_ids(self, text: str, language: str = "en") -> List[int]:
        phonemes = self.phonemize_text(text, language)
        phoneme_list = EmphasisModel.split_phonemes(phonemes)
        try:
            input_ids = phonemes_to_ids(phoneme_list)
        except Exception as e:
            print(f"Invalid phoneme found in: {phonemes}, error: {e}")
        return input_ids

    @staticmethod
    def preprocess_phonemes(phonemes: str) -> str:
        return EmphasisModel.split_phonemes(phonemes)

    @staticmethod
    def postprocess_prediction(phonemes: str, start_idx: int, end_idx: int) -> str:
        return ''.join(phonemes[:start_idx] + ['"'] + phonemes[start_idx:end_idx+1] + ['"'] + phonemes[end_idx+1:])
    
    @staticmethod
    def prepare_escaped_symbols():
        escaped_symbols = [re.escape(symbol) for symbol in IPA_LIST]
        escaped_symbols.sort(key=lambda x: -len(x))
        return '|'.join(escaped_symbols)

    @staticmethod
    def split_phonemes(input_string: str) -> List[str]:
        input_string = re.sub(r'\s+([,.;?!])', r'\1', input_string)
        return re.findall(EmphasisModel.escaped_symbols, input_string)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # remove spaces before punctuation
        text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
        return text

    def phonemize_text(self, text: str, language: str) -> str:
        text = self.preprocess_text(text)
        phonemes = []
        words = []
        in_quotes = False
        
        for sentence in sentences(text, lang=language):
            for word in sentence:
                if word.text == '"':
                    phonemes, words, in_quotes = self.handle_quote(phonemes, words, in_quotes)
                elif word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                elif word.phonemes:
                    phonemes, words = self.handle_word(phonemes, words, word, in_quotes)
        
        return ''.join(phonemes)

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
    