import logging
import json
import logging

from typing import Dict, Any

from .cosmos import Cosmos
from .tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, emmphasis_model_paths: Dict[str, str], db_paths: Dict[str, str], language: str = "en", use_cosmos: bool = False, cosmos_config: Dict[str, Any] = None):
        self.language=language
        self.emphasis_lookup = dict()
        self.cosmos_lookup = dict()
        
        if use_cosmos:
            if not cosmos_config:
                raise ValueError("Cosmos configuration is required when use_cosmos is True")
            self.cosmos_client = Cosmos(
                cosmos_config['url'],
                cosmos_config['key'],
                cosmos_config['database_name'],
                self.language
            )
            self.cosmos_lookup = self.load_emphasis_lookup_from_cosmos()
            logger.info(f"Loaded emphasis lookup from Cosmos DB for {self.language}: {self.cosmos_lookup}")

        if db_paths[language]:
            logger.info(f"Loading emphasis lookup from file: {db_paths[language]}")
            self.emphasis_lookup = self.load_emphasis_lookup_from_file(db_paths[language])
            
        self.emphasis_lookup.update(self.cosmos_lookup)
        self.tokenizer_manager = Tokenizer(emmphasis_model_paths, self.emphasis_lookup)
        self.tokenizer = self.tokenizer_manager.get_tokenizer(self.language)
        
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
        
    
    def get_input_ids(self, input_str: str, phonemes: bool = False, return_phonemes: bool = False, add_blank_token: bool = False, normalize: bool = False) -> dict:
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
            phonemes_str, normalized_text = self.tokenizer.phonemize_text(input_str, normalize=normalize)
        else:
            phonemes_str = input_str
        
        # Important that you split this do not return list of phonemes from self.tokenizer.phonemize_text, or there will be incorrect splitting.
        phoneme_list = self.tokenizer.split_phonemes(phonemes_str)
                
        if add_blank_token:
            phoneme_list.append(' ')
        
        if return_phonemes:
            result["phonemes"] = phonemes_str
        
        try:
            input_ids = self.tokenizer.phonemes_to_ids(phoneme_list)
            result["input_ids"] = input_ids
        except Exception as e:
            logger.error(f"Invalid phoneme found in: {phoneme_list}, error: {e}")
        
        return result

