import os
from text_processor import TextProcessor

def main():
    DATABASE_NAME = "Bookbot"
    
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
        "database_name": DATABASE_NAME
    }
    
    model = TextProcessor(model_dirs, db_paths, language="en", use_cosmos=True, cosmos_config=cosmos_config)
    
    # English Word input
    result = model.get_input_ids("Hello! my name is \"ladida\"....!", phonemes=False, return_phonemes=True, add_blank_token=True)
    print(result)
    
    # English Phoneme input
    phoneme = "hɛlˈoʊ mˈaɪ nˈeɪm ˈɪz"
    result = model.get_input_ids(phoneme, phonemes=True, return_phonemes=True, add_blank_token=True)
    print(result)
    
    # Swahili Word input
    model = TextProcessor(model_dirs, db_paths, language="sw", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("Jana nilitembelea mji wa \"Nairobi\". Niliona majengo \"marefu\" na magari mengi.", phonemes=False, return_phonemes=True, add_blank_token=True)
    print(result)
    
    # Indonesian Word input
    model = TextProcessor(model_dirs, db_paths, language="id", use_cosmos=False, cosmos_config=cosmos_config)
    result = model.get_input_ids("Halo nama saya Budi. Siapa \"nama\" kamu?", phonemes=False, return_phonemes=True, add_blank_token=True)
    print(result)

if __name__ == "__main__":
    main()
