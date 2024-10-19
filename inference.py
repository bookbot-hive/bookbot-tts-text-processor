import os
from phoneme_emphasis import EmphasisModel, Cosmos

URL = os.getenv("COSMOS_DB_URL")
KEY = os.getenv("COSMOS_DB_KEY")

def main():
    language = "en"
    DATABASE_NAME = "Bookbot"
    model_path = "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/google-cloud-functions/phoneme-emphasis/app/roberta-base-emphasis-onnx-quantized"
    db_path = "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/google-cloud-functions/phoneme-emphasis/app/db/words_emphasis_lookup_mixed.json"
    
    # move cosmos model initialization inside Emphasis Model
    cosmos = Cosmos(URL, KEY, DATABASE_NAME, language)
    # normalize_text = True will handle unicode normalziation, check out normalize.py
    model = EmphasisModel(model_path, db_path, language="en", cosmos_client=None, normalize_text=False)
    
    input_ids = model.get_input_ids("Hello! my name is \"bulubulu\"....!", phonemes=False, return_phonemes=True, add_blank_token=True)
    print(input_ids)
    phoneme = "hɛlˈoʊ mˈaɪ nˈeɪm ˈɪz"
    input_ids = model.get_input_ids(phoneme, phonemes=True, return_phonemes=True, add_blank_token=True)
    print(input_ids)
    
    model = EmphasisModel(model_path, db_path, cosmos_client=None, language="sw", normalize_text=False)
    input_ids = model.get_input_ids("Hujambo! Jina langu ni \"Bulubulu\"....!", phonemes=False, return_phonemes=True, add_blank_token=True)
if __name__ == "__main__":
    main()
