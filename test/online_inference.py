import os
from text_processor import TextProcessor
from pkg_resources import resource_filename


def main():
    model_dirs = {
        "en": "bookbot/roberta-base-emphasis-onnx-quantized",
        "sw": "",
        "id": "",
    }
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

    # ENGLISH
    # print("Initializing English model...")
    # model = TextProcessor(
    #     model_dirs["en"],
    #     db_paths["en"],
    #     language="en",
    #     use_cosmos=False,
    #     cosmos_config=cosmos_config,
    #     animation_tags_path="./animation_data.csv",
    #     online_g2p=True,
    # )
    # print("Model initialized, performing inference...")

    # result = model.get_input_ids(
    #     "<handRaiseHigh>.",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=False,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")

    # result = model.get_input_ids(
    #     "Hello World, my name is David! I'm a software engineer, and I love to code.",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=True,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")

    # result = model.get_input_ids(
    #     "Can you [lead] the <sound_part_showing_sound_1> conversation <smile>?",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=True,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}")

    # ### SWAHILI
    # model = TextProcessor(
    #     model_dirs["sw"],
    #     db_paths["sw"],
    #     language="sw",
    #     use_cosmos=False,
    #     cosmos_config=cosmos_config,
    #     animation_tags_path="./animation_data.csv",
    #     online_g2p=True,
    # )
    # result = model.get_input_ids(
    #     "Jana <handRaiseHigh> nilitembelea mji wa [Nairobi]. 4525 Niliona majengo [marefu] na magari mengi <sound_part_showing_sound_1>.",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=False,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")

    ### INDONESIAN
    model = TextProcessor(
        model_dirs["id"],
        db_paths["id"],
        language="id",
        use_cosmos=False,
        cosmos_config=cosmos_config,
        animation_tags_path="./animation_data.csv",
        online_g2p=True,
    )
    # result = model.get_input_ids(
    #     "<sound_scrolling_2> Halo <handRaiseHigh> nama saya Budi siapa [nama] kamu <sound_robotic_arm_2>?",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=False,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")
    # result = model.get_input_ids(
    #     "Ensiklopedia",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=False,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")

    # result = model.get_input_ids(
    #     "Daftar / Masuk. Non-Fiksi",
    #     phonemes=False,
    #     return_phonemes=True,
    #     push_oov_to_cosmos=False,
    #     add_blank_token=True,
    # )
    # print(f"Result: {result}\n")
    
    result = model.get_input_ids(
        "Minecraft",
        phonemes=False,
        return_phonemes=True,
        push_oov_to_cosmos=False,
        add_blank_token=True,
    )
    print(f"Result: {result}\n")
    result = model.get_input_ids(
        "minecraft",
        phonemes=False,
        return_phonemes=True,
        push_oov_to_cosmos=False,
        add_blank_token=True,
    )
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()
