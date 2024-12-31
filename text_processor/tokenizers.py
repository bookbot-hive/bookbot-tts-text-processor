import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from gruut import sentences
from g2p_id import G2p
from nltk.tokenize import sent_tokenize, TweetTokenizer
from functools import lru_cache
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count


from .symbols import SymbolSet
from .symbols import get_symbol_set
from .normalization import preprocess_text
from .utils import TextUtils


import numpy as np
import re
import logging
import uuid
import time
import os
import string 

TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, symbol_set: SymbolSet, online_g2p: bool = False):
        self.symbol_set = symbol_set
        self.language = language
        if emphasis_model_path:
            self.model, self.tokenizer = self.load_model_and_tokenizer(emphasis_model_path)
        if emphasis_lookup:
            self.emphasis_lookup = emphasis_lookup
        else:
            self.emphasis_lookup = {}
        self.escaped_symbols = self.prepare_escaped_symbols(symbol_set.SYMBOLS)
        self.escaped_symbols_pattern = re.compile(self.escaped_symbols)
        self.executor = ThreadPoolExecutor(max_workers=min(32, cpu_count() + 4))
        self.cosmos_client = None
        self.push_oov_to_cosmos = False
        self.turso_config = None
        if online_g2p:
            logger.info(f"Initializing online G2P for {language}")
            self._init_online_g2p(language)
            
       
        
    @abstractmethod
    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[List[str], str]:
        pass
    
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        return self.symbol_set.phonemes_to_ids(phonemes)
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        return self.symbol_set.ids_to_phonemes(ids)

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
        
    def _init_online_g2p(self, lang: str):
        if lang == "en":
            table = "en_phonemes"
        elif lang == "id":
            table = "id_phonemes"
        elif lang == "sw":
            table = "sw_phonemes"
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        self.turso_config = {
            "url": TURSO_URL,
            "auth_token": TURSO_AUTH_TOKEN,
            "table": table
        }
            
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
        tag_patterns = [f'<{tag}>' for tag in TextUtils.get_custom_tags().keys()]
        
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
        phoneme, _, _= self.phonemize_text(word, True)
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
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, online_g2p: bool = False):
        super().__init__(emphasis_model_path, emphasis_lookup, language, get_symbol_set("en"), online_g2p)
        self.accent = "en-us"  # Default accent
    
    def set_accent(self, accent: str):
        """Set the accent for English phonemization (en-us or en-gb)"""
        if accent not in ["en-us", "en-gb"]:
            raise ValueError("Accent must be either 'en-us' or 'en-gb'")
        self.accent = accent

    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[str, str, List[Tuple[int, int]]]:
        text = preprocess_text(text, normalize)
        phonemes = []
        word_boundaries = []
        current_pos = 0
        in_emphasis = False
        emphasized_words = []
        
        # Pre-process special tags with placeholders
        special_tags = []
        pattern = r'<[^>]+>'
        placeholder_template = 'TAGPLACEHOLDER{}'
        
        # Replace tags with placeholders and store original tags
        def replace_tag(match):
            tag = match.group(0)
            tag_name = tag.strip('<>')
            if tag_name in TextUtils.get_custom_tags():
                placeholder = placeholder_template.format(len(special_tags))
                special_tags.append(f'<{tag_name}>')
                return placeholder
            return tag

        processed_text = re.sub(pattern, replace_tag, text)
        
        # Use the specified accent for phonemization
        for sentence in sentences(processed_text, lang=self.accent, turso_config=self.turso_config):
            for word in sentence:
                if word.text.startswith('TAGPLACEHOLDER'):
                    # Handle special tags
                    try:
                        idx = int(word.text.replace('TAGPLACEHOLDER', ''))
                        tag = special_tags[idx]
                        phonemes.append(tag)
                        word_boundaries.append((current_pos, current_pos + len(tag)))
                        current_pos += len(tag)
                    except (IndexError, ValueError):
                        continue
                        
                elif word.text == '[':
                    in_emphasis = True
                    emphasized_words = []
                elif word.text == ']' and in_emphasis:
                    # Process all emphasized words as a single unit
                    start_pos = current_pos
                    trailing_punct = ""
                    
                    # Add a single space before the emphasized word if it's not the first word
                    if phonemes and phonemes[-1] != " ":
                        phonemes.append(" ")
                        current_pos += 1
                    
                    for i, emphasized_word in enumerate(emphasized_words):
                        if emphasized_word.text in [".", "!", "?", ",", ":", ";"]:
                            trailing_punct += emphasized_word.text
                        else:
                            emphasized_phonemes = self.handle_emphasized_word(emphasized_word)
                            logger.debug(f"Emphasized phonemes: {emphasized_phonemes}")
                            # Add space between each emphasized words
                            if i > 0:
                                phonemes.append(" ")
                                current_pos += 1
                            phonemes.append(emphasized_phonemes)
                            current_pos += len(emphasized_phonemes)
                    
                    if trailing_punct:
                        phonemes.append(trailing_punct)
                        current_pos += len(trailing_punct)
                    
                    # Add word boundary for the entire emphasized phrase
                    word_boundaries.append((start_pos, current_pos))
                    in_emphasis = False
                    
                elif in_emphasis:
                    emphasized_words.append(word)
                elif word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                    if word_boundaries:
                        word_boundaries[-1] = (word_boundaries[-1][0], current_pos + len(word.text))
                    else:
                        word_boundaries.append((current_pos, current_pos + len(word.text)))
                    current_pos += len(word.text)
                elif word.phonemes:
                    start_pos = current_pos
                    # Add a single space before non-emphasized words if needed
                    if phonemes and phonemes[-1] != " ":
                        phonemes.append(" ")
                        current_pos += 1
                    word_phonemes = self.handle_word([], word, in_emphasis)
                    phonemes.extend(word_phonemes)
                    current_pos += sum(len(p) for p in word_phonemes)
                    word_boundaries.append((start_pos, current_pos))
        
        return ''.join(phonemes), text, word_boundaries
    
    def handle_word(self, phonemes, word, in_emphasis):
        if not in_emphasis and phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        phonemes.append(''.join(word.phonemes))
        return phonemes
    
    def handle_emphasized_word(self, word):
        lookup_result = self.emphasis_lookup.get(word.text)
        emphasized_phonemes = None
        
        if isinstance(lookup_result, dict):
            # Handle homograph case
            if hasattr(word, 'pos') and word.pos in lookup_result:
                emphasized_phonemes = lookup_result[word.pos]
            else:
                # Default to first value if tag not found or no tag available
                emphasized_phonemes = next(iter(lookup_result.values()))
        else:
            # Handle normal case
            emphasized_phonemes = lookup_result
            
        if emphasized_phonemes is None:
            if hasattr(word, 'phonemes') and word.phonemes:
                emphasized_phonemes = self.emphasize_phonemes(''.join(word.phonemes))
                if self.push_oov_to_cosmos:
                    future = self.executor.submit(self._save_to_word_universal, word.text, emphasized_phonemes)
                    future.add_done_callback(self._handle_save_result)
            else:
                emphasized_phonemes = word.text
        
        return emphasized_phonemes
    
    def _handle_save_result(self, future):
        try:
            future.result()  # This will raise any exception that occurred during execution
        except Exception as e:
            logging.error(f"Error saving word to database: {e}")
            
    def __del__(self):
        # Ensure the executor is shut down when the object is destroyed
        self.executor.shutdown(wait=False)

class GruutSwahiliTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, online_g2p: bool = False):
        super().__init__(emphasis_model_path, emphasis_lookup, language, get_symbol_set("sw"), online_g2p)

    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[str, str, List[Tuple[int, int]]]:
        text = preprocess_text(text, normalize)
        phonemes = []
        word_boundaries = []
        current_pos = 0
        in_emphasis = False
        emphasized_words = []
        
        # Pre-process special tags with placeholders
        special_tags = []
        pattern = r'<[^>]+>'
        placeholder_template = 'TAGPLACEHOLDER{}'
        
        # Replace tags with placeholders and store original tags
        def replace_tag(match):
            tag = match.group(0)
            tag_name = tag.strip('<>')
            if tag_name in TextUtils.get_custom_tags():
                placeholder = placeholder_template.format(len(special_tags))
                special_tags.append(f'<{tag_name}>')
                return placeholder
            return tag

        processed_text = re.sub(pattern, replace_tag, text)
        
        for sentence in sentences(processed_text, lang="sw", turso_config=self.turso_config):
            for word in sentence:
                if word.text.startswith('TAGPLACEHOLDER'):
                    # Handle special tags
                    try:
                        idx = int(word.text.replace('TAGPLACEHOLDER', ''))
                        tag = special_tags[idx]
                        phonemes.append(tag)
                        word_boundaries.append((current_pos, current_pos + len(tag)))
                        current_pos += len(tag)
                    except (IndexError, ValueError):
                        continue
                        
                elif word.text == '[':
                    in_emphasis = True
                    emphasized_words = []
                elif word.text == ']' and in_emphasis:
                    # Process all emphasized words as a single unit
                    start_pos = current_pos
                    trailing_punct = ""
                    
                    # Add a single space before the emphasized word if it's not the first word
                    if phonemes and phonemes[-1] != " ":
                        phonemes.append(" ")
                        current_pos += 1
                    
                    for emphasized_word in emphasized_words:
                        if emphasized_word.text in [".", "!", "?", ",", ":", ";"]:
                            trailing_punct += emphasized_word.text
                        else:
                            emphasized_phonemes = self.handle_emphasized_word([], emphasized_word)
                            phonemes.extend(emphasized_phonemes)
                            current_pos += sum(len(p) for p in emphasized_phonemes)
                    
                    if trailing_punct:
                        phonemes.append(trailing_punct)
                        current_pos += len(trailing_punct)
                    
                    # Add word boundary for the entire emphasized phrase
                    word_boundaries.append((start_pos, current_pos))
                    in_emphasis = False
                    
                elif in_emphasis:
                    emphasized_words.append(word)
                elif word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                    if word_boundaries:
                        word_boundaries[-1] = (word_boundaries[-1][0], current_pos + len(word.text))
                    else:
                        word_boundaries.append((current_pos, current_pos + len(word.text)))
                    current_pos += len(word.text)
                elif word.phonemes:
                    start_pos = current_pos
                    # Add a single space before non-emphasized words if needed
                    if phonemes and phonemes[-1] != " ":
                        phonemes.append(" ")
                        current_pos += 1
                    word_phonemes = self.handle_word([], word, in_emphasis)
                    phonemes.extend(word_phonemes)
                    current_pos += sum(len(p) for p in word_phonemes)
                    word_boundaries.append((start_pos, current_pos))
        
        return ''.join(phonemes), text, word_boundaries

    def handle_word(self, phonemes, word, in_emphasis):
        if not in_emphasis and phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        phonemes.append(''.join(word.phonemes))
        return phonemes
    
    def handle_emphasized_word(self, phonemes, word):
        # [TODO] Add emphasis lookup and model for Swahili
        if phonemes and phonemes[-1] != ' ':
            phonemes.append(' ')
        phonemes.append(''.join(word.phonemes))
        return phonemes
    
    def _handle_save_result(self, future):
        try:
            future.result()  # This will raise any exception that occurred during execution
        except Exception as e:
            logging.error(f"Error saving word to database: {e}")
            
    def __del__(self):
        # Ensure the executor is shut down when the object is destroyed
        self.executor.shutdown(wait=False)

class G2pIdTokenizer(BaseTokenizer):
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, online_g2p: bool = False):
        super().__init__(emphasis_model_path, emphasis_lookup, language, get_symbol_set("id"), online_g2p)
        self.g2p = G2p(turso_config=self.turso_config)
        self.tokenizer = TweetTokenizer()
        self.puncts = ".,!?:"

    def phonemize_text(self, text: str, normalize: bool = False) -> Tuple[str, str, List[Tuple[int, int]]]:
            text = preprocess_text(text, normalize)
            phonemes = []
            word_boundaries = []
            current_pos = 0 
            
            # Pre-process special tags with placeholders
            special_tags = []
            pattern = r'<[^>]+>'
            placeholder_template = 'TAGPLACEHOLDER{}'
            
            # Replace tags with placeholders and store original tags
            def replace_tag(match):
                tag = match.group(0)
                tag_name = tag.strip('<>')
                if tag_name in TextUtils.get_custom_tags():
                    placeholder = placeholder_template.format(len(special_tags))
                    special_tags.append(f'<{tag_name}>')
                    return placeholder
                return tag

            processed_text = re.sub(pattern, replace_tag, text)
            sentences = sent_tokenize(processed_text)
            sentences = [sentence.replace("-", " ") for sentence in sentences]
            
            logger.debug(f"Sentences: {sentences}")
            
            for i, sentence in enumerate(sentences):
                words = []
                for word in sentence.split():
                    # Check if word ends with punctuation
                    if word and word[-1] in self.puncts:
                        words.append(word[:-1])  # Add word without punctuation
                        words.append(word[-1])   # Add punctuation as separate token
                    else:
                        words.append(word)
                logger.debug(f"Words: {words}")
                # Create a list of words to be processed by G2p
                g2p_words = []
                word_mapping = []  # Maps G2p results back to original word positions
                
                for idx, word in enumerate(words):
                    logger.debug(f"Word: {word}")
                    if not word.startswith('TAGPLACEHOLDER') and word not in self.puncts:
                        g2p_words.append(word)
                        word_mapping.append(idx)
                
                # Process regular words with G2p
                logger.debug(f"G2p words: {g2p_words}")
                sent_ph = self.g2p(' '.join(g2p_words)) if g2p_words else []
                logger.debug(f"Sent ph: {sent_ph}")
                # Process each word in original order
                ph_idx = 0
                
                for idx, word in enumerate(words):
                    if word.startswith('TAGPLACEHOLDER'):
                        try:
                            tag_idx = int(word.replace('TAGPLACEHOLDER', ''))
                            tag = special_tags[tag_idx]
                            start_pos = current_pos
                            phonemes.append(tag)
                            current_pos += len(tag)
                            word_boundaries.append((start_pos, current_pos))
                        except (IndexError, ValueError):
                            continue
                    elif word in self.puncts:
                        start_pos = current_pos
                        phonemes.append(word)
                        current_pos += len(word)
                        if word_boundaries:
                            word_boundaries[-1] = (word_boundaries[-1][0], current_pos)
                        else:
                            word_boundaries.append((start_pos, current_pos))
                    elif word in string.punctuation:
                        continue
                    else:
                        if phonemes and phonemes[-1] != " ":
                            phonemes.append(" ")
                            current_pos += 1
                        start_pos = current_pos
                        phoneme = ''.join(sent_ph[ph_idx])
                        phonemes.append(phoneme)
                        current_pos += len(phoneme)
                        word_boundaries.append((start_pos, current_pos))
                        ph_idx += 1
                        
            logger.debug(f"Phonemes: {phonemes}")
            logger.debug(f"Word boundaries: {word_boundaries}")
            return ''.join(phonemes), text, word_boundaries

class Tokenizer:
    def __init__(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, online_g2p: bool = False):
        self.tokenizer = self._create_tokenizer(emphasis_model_path, emphasis_lookup, language, online_g2p)

    def _create_tokenizer(self, emphasis_model_path: str, emphasis_lookup: Dict[str, str], language: str, online_g2p: bool = False) -> BaseTokenizer:
        if language == "en":
            return GruutTokenizer(emphasis_model_path, emphasis_lookup, language, online_g2p)
        elif language == "sw":
            return GruutSwahiliTokenizer(emphasis_model_path, emphasis_lookup, language, online_g2p)
        elif language == "id":
            return G2pIdTokenizer(emphasis_model_path, emphasis_lookup, language, online_g2p)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def get_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer
