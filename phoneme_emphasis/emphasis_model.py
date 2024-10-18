import logging
import re
import json
import logging
import numpy as np
from typing import List
from gruut import sentences
from transformers import PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForQuestionAnswering
from functools import lru_cache
from azure.cosmos import CosmosClient
from .utils import IPA_LIST
from .gruut_symbols import phonemes_to_ids
from .normalization import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmphasisModel:
    def __init__(self, model_dir: str, db_path: str, cosmos_client: CosmosClient = None, language: str = "en", normalize_text: bool = True):
        self.language = language
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_dir)
        self.cosmos_client = cosmos_client
        if cosmos_client:
            self.load_emphasis_lookup(db_path, True)
        else:
            self.load_emphasis_lookup(db_path)
        self.escaped_symbols = self.prepare_escaped_symbols()
        self.normalize_text = normalize_text
        
    def load_emphasis_lookup(self, db_path: str, cosmos_lookup: bool = False):
        with open(db_path, 'r') as f:
            self.emphasis_lookup = json.load(f)
            
        if cosmos_lookup:
            all_records = self.cosmos_client.get_all_records()
            wu_emphasis_dict = {
                record["word"]: record["emphasisIPA"] 
                for record in all_records 
                if "emphasisIPA" in record and record["emphasisIPA"].strip()
            }
            self.emphasis_lookup.update(wu_emphasis_dict)
            
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
    
    def get_input_ids(self, input_str: str, language: str = "en", phonemes: bool = False, return_phonemes: bool = False, add_blank_token: bool = False) -> dict:
        """
        Given a string of text or phonemes, return the input ids.

        Args:
            input_str (str): The text or phonemes to be converted to input ids.
            language (str, optional): The language of the text. Defaults to "en".
            phonemes (bool, optional): If True, input_str is treated as phonemes. Defaults to False.
            return_phonemes (bool, optional): If True, the phonemes will be returned. Defaults to False.
            add_blank_token (bool, optional): If True, a blank token will be added to the end of the phoneme list. Defaults to False.

        Returns:
            dict: A dictionary containing the input ids and optionally the phonemes.
        """
        result = {}
        
        if not phonemes:
            phonemes_str = self.phonemize_text(input_str, language)
        else:
            phonemes_str = input_str
        
        phoneme_list = self.split_phonemes(phonemes_str)
        if add_blank_token:
            phoneme_list.append(' ')
        
        if return_phonemes:
            result["phonemes"] = phonemes_str
        
        try:
            input_ids = phonemes_to_ids(phoneme_list)
            result["input_ids"] = input_ids
        except Exception as e:
            logger.error(f"Invalid phoneme found in: {phonemes_str}, error: {e}")
        
        return result
        
    def split_phonemes(self, input_string: str) -> List[str]:
        input_string = re.sub(r'\s+([,.;?!])', r'\1', input_string)
        return re.findall(self.escaped_symbols, input_string)

    @staticmethod
    def postprocess_prediction(phonemes: str, start_idx: int, end_idx: int) -> str:
        return ''.join(phonemes[:start_idx] + ['"'] + phonemes[start_idx:end_idx+1] + ['"'] + phonemes[end_idx+1:])
    
    @staticmethod
    def prepare_escaped_symbols():
        escaped_symbols = [re.escape(symbol) for symbol in IPA_LIST]
        escaped_symbols.sort(key=lambda x: -len(x))
        return '|'.join(escaped_symbols)

    def phonemize_text(self, text: str, language: str) -> str:   
        text = preprocess_text(text, normalize=self.normalize_text)
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
    
    
