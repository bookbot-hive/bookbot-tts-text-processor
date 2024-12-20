import os
from text_processor import TextProcessor
from pkg_resources import resource_filename


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

def generate_inputids(text, language):
    # If not planning to use emphasis then you can set use_cosmos=False and push_oov_to_cosmos=False
    model = TextProcessor(model_dirs[language], db_paths[language], language=language, use_cosmos=False, cosmos_config=cosmos_config) 
    return model, model.get_input_ids(text, phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)

if __name__ == "__main__":
    model, result = generate_inputids("Bookbot uses speech recognition. Please enable the microphone while using this app.", "en")
    print(result)
    print(model.tokenizer.split_phonemes(model.tokenizer.ids_to_phonemes(result['input_ids'])))
    