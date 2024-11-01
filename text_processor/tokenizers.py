from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from gruut import sentences
from g2p_id import G2p
from nltk.tokenize import sent_tokenize, TweetTokenizer
from functools import lru_cache
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ThreadPoolExecutor

from . import gruut_symbols
from . import gruut_sw_symbols
from . import g2p_id_symbols
from .normalization import preprocess_text

import numpy as np
import re
import logging
import uuid
import time

from .utils import CUSTOM_TAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, symbols: List[str]):
        self.language = language
        if emphasis_model_path:
            self.model, self.tokenizer = self.load_model_and_tokenizer(emphasis_model_path)
        if emphasis_lookup:
            self.emphasis_lookup = emphasis_lookup
        self.escaped_symbols = self.prepare_escaped_symbols(symbols)
        self.escaped_symbols_pattern = re.compile(self.escaped_symbols)
        # logger.info(f"escaped_symbols: {self.escaped_symbols}")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.cosmos_client = None
        self.push_oov_to_cosmos = False
        
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
        model = ORTModelForQuestionAnswering.from_pretrained(
            model_dir,
            providers=['CPUExecutionProvider']
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        return model, tokenizer

    def set_cosmos_client(self, cosmos_client):
        self.cosmos_client = cosmos_client
    
    def set_push_oov_to_cosmos(self, push_oov_to_cosmos: bool):
        self.push_oov_to_cosmos = push_oov_to_cosmos

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

    # def split_phonemes(self, input_string: str) -> List[str]:
    #     # remove spaces before end of punctuations
    #     return re.findall(self.escaped_symbols, input_string)
    
    def split_phonemes(self, input_string: str) -> List[str]:
        logger.debug(f"input_string: {input_string}")
        word_phonemes = input_string.split(" ")
        
        result = []
        for i, word_ipa in enumerate(word_phonemes):
            result.extend(self.escaped_symbols_pattern.findall(word_ipa))
            if i < len(word_phonemes) - 1:
                result.append(' ')
        
        logger.debug(f"split_phonemes result: {result}")
        return result

    @staticmethod
    def postprocess_prediction(phonemes: str, start_idx: int, end_idx: int) -> str:
        return ''.join(phonemes[:start_idx] + ['"'] + phonemes[start_idx:end_idx+1] + ['"'] + phonemes[end_idx+1:])

    def prepare_escaped_symbols(self, symbols: List[str]):
        # Sort symbols by length (longest first) to ensure correct matching
        sorted_symbols = sorted(symbols, key=len, reverse=True)
        
        # Add custom tags to the pattern
        tag_patterns = [f'<{tag}>' for tag in CUSTOM_TAGS.keys()]
        
        # Combine symbols and tags, escape special regex characters
        all_patterns = sorted_symbols + tag_patterns
        escaped_patterns = [re.escape(s) for s in all_patterns]
        
        # Join all patterns with | for alternation
        return '|'.join(escaped_patterns)
    
    def _save_to_word_universal(self, word: str, emphasized_phonemes: str):
        word_item = self._create_word_item(word, emphasized_phonemes)
        logger.info(f"New word record: {word_item}")
        self.cosmos_client.word_universal_container.upsert_item(word_item)
        logger.info(f"Saved new word record for '{word}' with emphasis '{emphasized_phonemes}'")
        
    def _create_word_item(self, word: str, emphasized_phonemes: str) -> dict:
        timestamp = round(time.time() * 1000)
        phoneme, _ = self.phonemize_text(word, True)
        return {
            "id": str(uuid.uuid4()),
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "ipa": phoneme,
            "emphasisIPA": emphasized_phonemes,
            "language": self.language,
            "word": word.lower(),
            "syllable": "",
            "level": 1,
            "multiWord": "",
            "lexicons": [],
            "pos": "",
            "validated": False,
            "inUse": False,
            "partition": "default",
        }
    
class GruutTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str):
        super().__init__(emphasis_model_path, emphasis_lookup, language, gruut_symbols.SYMBOLS)
    
    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[List[str], str]:
        text = preprocess_text(text, normalize)
        phonemes = []
        words = []
        in_emphasis = False
        in_tag = False
        current_tag = []
        emphasized_words = []
        
        for sentence in sentences(text, lang="en"):
            for word in sentence:
                if word.text.startswith('<'):
                    in_tag = True
                    current_tag.append(word.text)
                    continue
                elif word.text.endswith('>') and in_tag:
                    current_tag.append(word.text)
                    # if phonemes and phonemes[-1] != ' ':
                    #     phonemes.append(' ')
                    phonemes.append(''.join(current_tag))
                    current_tag = []
                    in_tag = False
                    continue
                elif in_tag:
                    current_tag.append(word.text)
                    continue
                    
                if word.text == '[':
                    in_emphasis = True
                    emphasized_words = []
                elif word.text == ']' and in_emphasis:
                    trailing_punct = ""
                    for emphasized_word in emphasized_words:
                        if emphasized_word.text in [".", "!", "?", ",", ":", ";"]:
                            trailing_punct += emphasized_word.text
                        else:
                            phonemes, words = self.handle_emphasized_word(phonemes, words, emphasized_word)
                    if trailing_punct:
                        phonemes.append(trailing_punct)
                    in_emphasis = False
                elif in_emphasis:
                    emphasized_words.append(word) 
                elif word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                elif word.phonemes:
                    phonemes, words = self.handle_word(phonemes, words, word, in_emphasis)
        
        return ''.join(phonemes), text

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        return gruut_symbols.phonemes_to_ids(phonemes)
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        return gruut_symbols.ids_to_phonemes(ids)
    
    def handle_word(self, phonemes, words, word, in_emphasis):
        if not in_emphasis and phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        phonemes.append(''.join(word.phonemes))
        if in_emphasis:
            words.append(word.text)
        return phonemes, words
    
    def handle_emphasized_word(self, phonemes, words, word):
        if phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        emphasized_phonemes = self.emphasis_lookup.get(word.text)
        if emphasized_phonemes is None:
            if hasattr(word, 'phonemes') and word.phonemes:
                emphasized_phonemes = self.emphasize_phonemes(''.join(word.phonemes))
                if self.push_oov_to_cosmos:
                    future = self.executor.submit(self._save_to_word_universal, word.text, emphasized_phonemes)
                    future.add_done_callback(self._handle_save_result)
            else:
                emphasized_phonemes = word.text
                
        phonemes.append(emphasized_phonemes)
        
        return phonemes, words
    
    def _handle_save_result(self, future):
        try:
            future.result()  # This will raise any exception that occurred during execution
        except Exception as e:
            logging.error(f"Error saving word to database: {e}")
            
    def __del__(self):
        # Ensure the executor is shut down when the object is destroyed
        self.executor.shutdown(wait=False)

class GruutSwahiliTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str):
        super().__init__(emphasis_model_path, emphasis_lookup, language, gruut_sw_symbols.SYMBOLS)
        
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
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str):
        super().__init__(emphasis_model_path, emphasis_lookup, language, g2p_id_symbols.SYMBOLS)
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
                if word == '[':
                    sent_ph.insert(idx, '[')
                elif word == ']':
                    sent_ph.insert(idx, ']')
                    
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
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str):
        self.tokenizer = self._create_tokenizer(emphasis_model_path, emphasis_lookup, language)

    def _create_tokenizer(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str) -> BaseTokenizer:
        if language == "en":
            return GruutTokenizer(emphasis_model_path, emphasis_lookup, language)
        elif language == "sw":
            return GruutSwahiliTokenizer(emphasis_model_path, emphasis_lookup, language)
        elif language == "id":
            return G2pIdTokenizer(emphasis_model_path, emphasis_lookup, language)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def get_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer
