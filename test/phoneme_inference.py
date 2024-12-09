from text_processor import TextProcessor

def main():
    text_processor = TextProcessor({}, {}, language="en", use_cosmos=False,  animation_tags_path="data.csv")
    # Set add_blank_token to True to add a blank token at the end of the text.
    result = text_processor.get_input_ids("<sound_bell_sound> <hapticMedium> Great!", return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()