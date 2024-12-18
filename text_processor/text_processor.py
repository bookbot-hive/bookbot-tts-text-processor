import logging
import json
import logging

from typing import Dict, Any

from .cosmos import Cosmos
from .tokenizers import Tokenizer
from .utils import TextUtils
from .prompt import PROMPT
from .gpt import GPT
from .claude import Claude

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, emphasis_model_path: str, db_path: str, language: str = "en", use_cosmos: bool = False, cosmos_config: Dict[str, Any] = None, animation_tags_path: str = None, online_g2p: bool = False):
        self.language = language
        self.emphasis_lookup = dict()
        self.cosmos_lookup = dict()
        
        # Initialize custom tags
        TextUtils.initialize_custom_tags(animation_tags_path)
        logger.info(f"Loaded custom animation tags from: {animation_tags_path}")
        
        if use_cosmos:
            if not cosmos_config:
                raise ValueError("Cosmos configuration is required when use_cosmos is True")
            self.cosmos_client = Cosmos(
                cosmos_config['url'],
                cosmos_config['key'],
                cosmos_config['database_name'],
                self.language
            )
            self.emphasis_lookup = self.load_emphasis_lookup_from_cosmos()
            logger.info(f"Loaded emphasis lookup from Cosmos DB for {self.language}")
        elif db_path:
            logger.info(f"Loading emphasis lookup from file: {db_path}")
            self.emphasis_lookup = self.load_emphasis_lookup_from_file(db_path)
            
        self.tokenizer_manager = Tokenizer(emphasis_model_path, self.emphasis_lookup, self.language, online_g2p=online_g2p)
        self.tokenizer = self.tokenizer_manager.get_tokenizer()
        if use_cosmos:
            self.tokenizer.set_cosmos_client(self.cosmos_client)
            
        if self.language == "en":
            self.tokenizer.set_accent("en-us")

    def load_emphasis_lookup_from_file(self, db_path: str):
        with open(db_path, 'r') as f:
            emphasis_lookup = json.load(f)
        return emphasis_lookup
        
    def load_emphasis_lookup_from_cosmos(self):
        all_records = self.cosmos_client.get_all_records()
        wu_emphasis_dict = {
        record["word"]: record["emphasisIPA"] 
            for record in all_records 
            if "emphasisIPA" in record and record["emphasisIPA"].strip()
        }
        return wu_emphasis_dict
        
    
    def get_input_ids(self, input_str: str, phonemes: bool = False, return_phonemes: bool = False, push_oov_to_cosmos: bool = False, add_blank_token: bool = False, normalize: bool = False, accent: str = None, emphasize_model: str = None) -> dict:
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
        
        if self.tokenizer.cosmos_client and push_oov_to_cosmos:
            self.tokenizer.set_push_oov_to_cosmos(True)
        else:
            self.tokenizer.set_push_oov_to_cosmos(False)
        
        try:
            if emphasize_model:
                # Check if input_str is a single word (no spaces)
                if ' ' not in input_str.strip():
                    input_str = f"[{input_str.strip()}]"
                else:
                    if "claude" in emphasize_model.lower():
                        input_str = Claude(model=emphasize_model).emphasize(PROMPT, input_str)
                    elif "gpt" in emphasize_model.lower():
                        input_str = GPT(model=emphasize_model).emphasize(PROMPT, input_str)
                    
                logger.info(f"Emphasized text: {input_str}")
        except Exception as e:
            logger.error(f"Error emphasizing text: {input_str}", exc_info=True)
            raise ValueError(f"Failed to emphasize text: {str(e)}")

        # set accent if provided
        if self.language == "en" and accent:
            self.tokenizer.set_accent(accent)
            
        try:
            if not phonemes:
                logger.debug(f"Converting text to phonemes: {input_str}")
                phonemes_str, normalized_text, word_boundaries = self.tokenizer.phonemize_text(input_str, normalize=normalize)
                result["normalized_text"] = normalized_text
            else:
                phonemes_str = input_str
                word_boundaries = [(0, len(input_str))]
                logger.debug(f"Using provided phonemes: {phonemes_str}")
            if return_phonemes:
                result["phonemes"] = phonemes_str
        except Exception as e:
            logger.error(f"Error converting text to phonemes: {input_str}", exc_info=True)
            raise ValueError(f"Failed to convert text to phonemes: {str(e)}")
        
        try:
            phoneme_list = self.tokenizer.split_phonemes(phonemes_str)
            if add_blank_token:
                phoneme_list.append(' ')
            logger.debug(f"Split phonemes: {phoneme_list}")
        except Exception as e:
            logger.error(f"Error splitting phonemes: {phonemes_str}", exc_info=True)
            raise ValueError(f"Failed to split phonemes: {str(e)}")
        
        try:
            input_ids = self.tokenizer.phonemes_to_ids(phoneme_list)
            result["input_ids"] = input_ids
            
            # Generate word_idx based on word boundaries
            word_idx = []
            current_pos = 0
            current_word = 0
            
            for phoneme in phoneme_list:
                # Find which word boundary this phoneme belongs to
                while current_word < len(word_boundaries) and current_pos >= word_boundaries[current_word][1]:
                    current_word += 1
                    
                # If this is a space, use previous word's index
                if phoneme == ' ':
                    word_idx.append(max(0, current_word - 1))
                else:
                    word_idx.append(current_word)
                    
                current_pos += len(phoneme)
            
            result["word_idx"] = word_idx
            logger.debug(f"Successfully generated input_ids and word_idx for: {input_str}")
            
        except Exception as e:
            logger.error(f"Failed to convert phonemes to IDs. Input: {input_str}", exc_info=True)
            logger.error(f"Problematic phoneme list: {phoneme_list}")
            raise ValueError(f"Failed to process phonemes: {str(e)}")
        
        return result

