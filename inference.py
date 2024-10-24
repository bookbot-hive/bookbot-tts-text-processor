import os
import time
from text_processor import TextProcessor

def main():    
    model_dirs = {
        "en": "bookbot/roberta-base-emphasis-onnx-quantized",
        "sw": "",
        "id": ""
    }
    db_paths = {
        "en": "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/Emphasizer/data/words_emphasis_lookup_mixed.json",
        "sw": "",
        "id": ""
    }
    
    cosmos_config = {
        "url": os.getenv("COSMOS_DB_URL"),
        "key": os.getenv("COSMOS_DB_KEY"),
        "database_name": "Bookbot"
    }

    start_time = time.time()
    # English
    model = TextProcessor(model_dirs["en"], db_paths["en"], language="en", use_cosmos=False, cosmos_config=cosmos_config)
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    
    result = model.get_input_ids("Hello <wave> world <listen> how are you? <headLean>", phonemes=False, return_phonemes=True, push_oov_to_cosmos=True, add_blank_token=True)
    print(result)
    
    
    # English Word input
    start_time = time.time()
    result = model.get_input_ids("Hello! my name is \"ladidadid\"....!", phonemes=False, return_phonemes=True, push_oov_to_cosmos=True, add_blank_token=True)
    end_time = time.time()
    print(f"Time taken to process word input: {end_time - start_time} seconds")
    print(result)
    
    # English Phoneme input
    phoneme = "hɛlˈoʊ mˈaɪ nˈeɪm ˈɪz"
    result = model.get_input_ids(phoneme, phonemes=True, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(result)
    
    # Swahili Word input
    model = TextProcessor(model_dirs["sw"], db_paths["sw"], language="sw", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("Jana nilitembelea mji wa \"Nairobi\". Niliona majengo \"marefu\" na magari mengi.", phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(result)
    
    # Indonesian Word input
    model = TextProcessor(model_dirs["id"], db_paths["id"], language="id", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("Halo nama saya Budi. Siapa \"nama\" kamu?", phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(result)

if __name__ == "__main__":
    main()