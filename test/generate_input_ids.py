import os
from text_processor import TextProcessor
from pkg_resources import resource_filename
import argparse
import json
import re
from datetime import datetime

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

models = {}


def generate_inputids(text, language, anim_path):
    # If not planning to use emphasis then you can set use_cosmos=False and push_oov_to_cosmos=False
    if language not in models:
        models[language] = TextProcessor(
            model_dirs[language],
            db_paths[language],
            language=language,
            use_cosmos=False,
            cosmos_config=cosmos_config,
            animation_tags_path=anim_path,
        )
    model = models[language]
    return model.get_input_ids(
        text,
        phonemes=False,
        return_phonemes=True,
        push_oov_to_cosmos=False,
        add_blank_token=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path", "-i", type=str, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_json_path", "-o", type=str, help="Path to the output JSON file"
    )
    parser.add_argument(
        "--anim_path", "-a", type=str, help="Path to animation CSV file"
    )
    parser.add_argument(
        "--log_path", "-l", type=str, help="Path to store error logs", default=None
    )
    args = parser.parse_args()

    # Create error log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_path:
        log_dir = args.log_path
    else:
        log_dir = os.path.dirname(args.output_json_path)
    os.makedirs(log_dir, exist_ok=True)

    error_log_path = os.path.join(log_dir, f"error_log_{timestamp}.txt")

    output_data = []
    if os.path.exists(args.output_json_path):
        print(f"Output file {args.output_json_path} exists, loading existing data...")
        with open(args.output_json_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        # Create set of existing hashes for efficient lookup
        existing_hashes = {item["hash"] for item in output_data}

        # Read the input JSON file
        with open(args.input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Filter for only new items
        new_items = [
            item for item in input_data if item.get("hash") not in existing_hashes
        ]
        print(f"Found {len(new_items)} new items to process")

        error_list = []
        anim_path = args.anim_path

        # Process only new items
        for item in new_items:
            original_text = item.get("text", "")
            language = item.get("language", "")
            hash_value = item.get("hash", "")
            try:
                # Remove any substrings enclosed in "{}"
                text = re.sub(r"\{.*?\}", "", original_text)

                # Replace "[" with "<" and "]" with ">"
                text = text.replace("[", "<").replace("]", ">")

                # Now process the cleaned text
                result = generate_inputids(text, language, anim_path)
                input_ids = result["input_ids"]
                phonemes = result["phonemes"]
                wordIdx = result["word_idx"]

                # Build the output item
                output_item = {
                    "hash": hash_value,
                    "inputIds": input_ids,
                    "language": language,
                    "phonemes": phonemes,
                    "wordIndexes": wordIdx,
                }

                print(
                    f"Processed item: hash '{hash_value}', language '{language}', input_ids '{input_ids}'"
                )

                # Append new items to existing output_data
                output_data.append(output_item)

            except Exception as e:
                error_message = f"Item '{hash_value}', language '{language}', text '{original_text}', errordetails: {e}"
                print(error_message)
                error_list.append(error_message)

    else:
        # Read the input JSON file
        with open(args.input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        output_data = []
        anim_path = args.anim_path
        error_list = []

        # Process items and collect errors
        for item in input_data:
            original_text = item.get("text", "")
            language = item.get("language", "")
            hash_value = item.get("hash", "")
            try:
                # Remove any substrings enclosed in "{}"
                text = re.sub(r"\{.*?\}", "", original_text)

                # Replace "[" with "<" and "]" with ">"
                text = text.replace("[", "<").replace("]", ">")

                # Now process the cleaned text
                result = generate_inputids(text, language, anim_path)
                input_ids = result["input_ids"]
                phonemes = result["phonemes"]
                wordIdx = result["word_idx"]

                # Build the output item
                output_item = {
                    "hash": hash_value,
                    "inputIds": input_ids,
                    "language": language,
                    "phonemes": phonemes,
                    "wordIndexes": wordIdx,
                }

                print(
                    f"Processed item: hash '{hash_value}', language '{language}', input_ids '{input_ids}'"
                )

                # Append the result to the output data list
                output_data.append(output_item)

            except Exception as e:
                error_message = f"Item '{hash_value}', language '{language}', text '{original_text}', errordetails: {e}"
                print(error_message)
                error_list.append(error_message)

    # Save the combined output data
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Write errors to log file if any occurred
    if error_list:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("The following errors occurred during processing:\n\n")
            f.write("\n\n".join(error_list))
        print(f"Errors were logged to: {error_log_path}")
        raise Exception(
            f"Processing completed with errors. See {error_log_path} for details"
        )

    print(f"Processing completed successfully. Output saved to {args.output_json_path}")
