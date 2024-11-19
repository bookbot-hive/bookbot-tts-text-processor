import os
import time
from text_processor import TextProcessor
from pkg_resources import resource_filename


def main():    
    model_dirs = {
        "en": "bookbot/roberta-base-emphasis-onnx-quantized",
        "sw": "",
        "id": ""
    }
    db_paths = {
        "en": resource_filename('text_processor', 'data/en_word_emphasis_lookup_mix_homographs.json'),
        "sw": "",
        "id": ""
    }
    
    cosmos_config = {
        "url": os.getenv("COSMOS_DB_URL"),
        "key": os.getenv("COSMOS_DB_KEY"),
        "database_name": "Bookbot"
    }

    # English
    model = TextProcessor(model_dirs["en"], db_paths["en"], language="en", use_cosmos=False, cosmos_config=cosmos_config)
    
    # English Word input
    result = model.get_input_ids("Hello <wave> world <listen> how are [you]? <headLean>", phonemes=False, return_phonemes=True, push_oov_to_cosmos=True, add_blank_token=True)
    print(f"Result: {result}")
    
    result = model.get_input_ids("Can you [lead] the <nod> conversation <smile>?", phonemes=False, return_phonemes=True, push_oov_to_cosmos=True, add_blank_token=True)
    print(f"Result: {result}")
    
    result = model.get_input_ids("I am 10 years old", phonemes=False, return_phonemes=True, push_oov_to_cosmos=True, add_blank_token=True)
    print(f"Result: {result}")
    
    # English Phoneme input
    phoneme = "hɛlˈoʊ mˈaɪ nˈeɪm ˈɪz"
    result = model.get_input_ids(phoneme, phonemes=True, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(f"Result: {result}")
    
    # # Swahili Word input
    model = TextProcessor(model_dirs["sw"], db_paths["sw"], language="sw", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("Jana <handRaiseHigh> nilitembelea mji wa [Nairobi]. 4525 Niliona majengo [marefu] na magari mengi.", phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(f"Result: {result}")
    
    # # Indonesian Word input
    model = TextProcessor(model_dirs["id"], db_paths["id"], language="id", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("<poked> Halo <handRaiseHigh> nama saya Budi siapa [nama] kamu <wave>?", phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    result = model.tokenizer.ids_to_phonemes(result["input_ids"])
    print(f"Result: {result}")

if __name__ == "__main__":
    main()