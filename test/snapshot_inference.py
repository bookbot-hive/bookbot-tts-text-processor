import os
from text_processor import TextProcessor
from pkg_resources import resource_filename


model_dirs = {"en": "bookbot/roberta-base-emphasis-onnx-quantized", "sw": "", "id": ""}

db_paths = {
    "en": resource_filename(
        "text_processor", "data/en_word_emphasis_lookup_mix_homographs.json"
    ),
    "sw": "",
    "id": "",
}

cosmos_config = {
    "url": os.getenv("COSMOS_DB_URL"),
    "key": os.getenv("COSMOS_DB_KEY"),
    "database_name": "Bookbot",
}


def generate_inputids(text, language):
    # If not planning to use emphasis then you can set use_cosmos=False and push_oov_to_cosmos=False
    model = TextProcessor(
        model_dirs[language],
        db_paths[language],
        language=language,
        use_cosmos=True,
        cosmos_config=cosmos_config,
    )
    return model.get_input_ids(
        text,
        phonemes=False,
        return_phonemes=True,
        push_oov_to_cosmos=True,
        add_blank_token=True,
        normalize=True,
    )


if __name__ == "__main__":
    print(generate_inputids("a.", "en"))
    print(generate_inputids("<handRaiseMid>.", "en"))
    print(generate_inputids("What do you think, Lachlan?", "en"))
    print(generate_inputids("Halo nama saya Budi. Siapa [nama] kamu?", "id"))
    print(generate_inputids("Ke.", "id"))
