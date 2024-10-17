import torch
import logging
import re
import json
from typing import List
from gruut import sentences
from transformers import PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForQuestionAnswering
from .utils import IPA_LIST
from .gruut_symbols import phonemes_to_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmphasisModel:
    def __init__(self, model_dir: str, db_path: str):
        self.device = torch.device("cpu")
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_dir)
        self.load_emphasis_lookup(db_path)
        
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
        emphasized = phonemes[:start_idx] + ['"'] + phonemes[start_idx:end_idx+1] + ['"'] + phonemes[end_idx+1:]
        return ''.join(emphasized)

    @staticmethod
    def split_phonemes(input_string: str) -> List[str]:
        input_string = re.sub(r'\s+([,.;?!])', r'\1', input_string)
        escaped_symbols = [re.escape(symbol) for symbol in IPA_LIST]
        escaped_symbols.sort(key=lambda x: -len(x))
        pattern = '|'.join(escaped_symbols)
        return re.findall(pattern, input_string)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # remove spaces before punctuation
        text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
        return text

    def phonemize_text(self, text: str, language: str) -> str:
        text = EmphasisModel.preprocess_text(text)
        phonemes = []
        words = []
        in_quotes = False
        for sentence in sentences(text, lang=language):
            for word in sentence:
                if word.text == '"':
                    # reach second double quote which indicate the end of an emphasized word
                    if in_quotes and words:
                        try:
                            emphasized_phonemes = self.emphasis_lookup[words[-1]]
                            # if emphasized phoneme exist in database pop the last phoneme and the double quote before it
                            phonemes.pop()
                            phonemes.pop()
                            words.pop()
                            phonemes.append(emphasized_phonemes)
                        except KeyError:
                            # if doesn't exist in database, emphasize with the transformer model
                            emphasized = self.emphasize_phonemes(phonemes[-1])
                            phonemes.pop()
                            phonemes.pop()
                            words.pop()
                            phonemes.append(emphasized)
                        in_quotes = False
                    # reach first double quote which indicate the start of an emphasized word
                    else:
                        # add space before the word if there is no space after the last phoneme
                        if phonemes and phonemes[-1] != ' ':
                            phonemes.append(' ')
                        phonemes.append('"')
                        in_quotes = True
                elif word.is_major_break or word.is_minor_break:
                    # if not in_quotes and phonemes and phonemes[-1] != ' ':
                    #     phonemes.append(' ')
                    phonemes.append(word.text)
                elif word.phonemes:
                    # add spaces in between words
                    if not in_quotes and phonemes and phonemes[-1] != ' ':
                        phonemes.append(' ')
                    phonemes.append(''.join(word.phonemes))
                    if in_quotes:
                        words.append(word.text)
        
        return ''.join(phonemes)
    