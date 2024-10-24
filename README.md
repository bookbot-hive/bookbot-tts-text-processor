# Optispeech Text Processor

This project implements a multi-language text processor for Bookbot Optispeech TTS, it can handle English, Swahili, and Indonesian inputs. It uses the `TextProcessor` class to process text and generate input IDs for various language models.

## Installation

To install the package, run:
`pip install git+https://github.com/bookbot-hive/bookbot-tts-text-processor.git`

To install with a specific version, run:
`pip install git+https://github.com/bookbot-hive/bookbot-tts-text-processor.git@v0.0.0`

after running the above command, make sure that gruut[sw] is installed if it's not, install it manually by running:
`pip install -f 'https://synesthesiam.github.io/prebuilt-apps/' gruut[sw]`


## Building the Package

If you want to build the package from source, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/bookbot-hive/bookbot-tts-text-processor.git
   cd text_processor
   ```

2. Install the package:
   ```
   pip install .
   ```

   Or, if you want to install it in editable mode for development:
   ```
   pip install -e .
   ```

_t## Usage

Here's a basic example of how to use the `TextProcessor`:

```python
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

# English
model = TextProcessor(model_dirs["en"], db_paths["en"], language="en", use_cosmos=True, cosmos_config=cosmos_config)

# English Word input
result = model.get_input_ids("Hello! my name is \"ladida\"...!", phonemes=False, return_phonemes=True, push_oov_to_cosmos=False, add_blank_token=True)
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
```

Output:
```
{'phonemes': 'hɛlˈoʊ! mˈaɪ nˈeɪm ˈɪz "lˈæd"ɪdə....!', 'input_ids': [23, 47, 27, 59, 4, 3, 28, 55, 3, 29, 57, 28, 3, 67, 38, 3, 5, 27, 61, 19, 5, 49, 19, 45, 12, 12, 12, 12, 4, 3]}
{'phonemes': 'hɛlˈoʊ mˈaɪ nˈeɪm ˈɪz', 'input_ids': [23, 47, 27, 59, 3, 28, 55, 3, 29, 57, 28, 3, 67, 38, 3]}
{'phonemes': 'ʄɑnɑ nilitɛᵐɓɛlɛɑ mʄi wɑ "nɑiɾɔɓi". niliɔnɑ mɑʄɛᵑgɔ "mɑɾɛfu" nɑ mɑɠɑɾi mɛᵑgi.', 'input_ids': [44, 35, 23, 35, 3, 23, 18, 21, 18, 26, 39, 46, 39, 21, 39, 35, 3, 22, 44, 18, 3, 30, 35, 3, 5, 23, 35, 18, 42, 37, 36, 18, 5, 12, 3, 23, 18, 21, 18, 37, 23, 35, 3, 22, 35, 44, 39, 47, 37, 3, 5, 22, 35, 42, 39, 16, 28, 5, 3, 23, 35, 3, 22, 35, 40, 35, 42, 18, 3, 22, 39, 47, 18, 12, 3]}
{'phonemes': 'halo nama saja budi. siapa "nama" kamu?', 'input_ids': [23, 16, 27, 30, 3, 29, 16, 28, 16, 3, 33, 16, 38, 16, 3, 17, 35, 19, 24, 12, 3, 33, 24, 16, 31, 16, 3, 5, 29, 16, 28, 16, 5, 3, 26, 16, 28, 35, 15, 3]}
```

## Parameters

### TextProcessor Initialization

- `emphasis_model_path`: Path to the emphasis model directory.
- `db_path`: Path to the database file for word emphasis lookup.
- `language`: The language to use (default is "en" for English).
- `use_cosmos`: Boolean flag to use emphasisIPA from Azure Cosmos DB (default is False).
- `cosmos_config`: Configuration dictionary for Azure Cosmos DB connection.

### get_input_ids Method

- `text`: The input text to process.
- `phonemes`: Boolean flag indicating if the input is phonemes (default is False).
- `return_phonemes`: Boolean flag to return phonemes (default is True).
- `push_oov_to_cosmos`: Boolean flag to push out-of-vocabulary (OOV) emphasis to Cosmos DB (default is False), to enable this you must set `use_cosmos=True` during initialization. 
- `add_blank_token`: Boolean flag to add blank tokens at the end of the input_ids (default is True).
- `normalize`: Boolean flag to normalize the input text (default is False).

## Supported Languages

- English (en)
- Swahili (sw)
- Indonesian (id)

## Azure Cosmos DB Integration

This project can optionally integrate with Azure Cosmos DB. To use this feature, set `use_cosmos=True` and provide the necessary configuration in `cosmos_config`.

## Environment Variables

The following environment variables are used:

- `COSMOS_DB_URL`: The URL for your Azure Cosmos DB instance.
- `COSMOS_DB_KEY`: The access key for your Azure Cosmos DB instance.

## TO DO's

- [ ] Add Swahili and Indonesian emphasis models to the project
- [ ] Add Swahili and Indonesian word to phoneme emphasis lookup.
- [ ] Support body language tags handling for Swahili and Indonesian

## License

This project is licensed under the MIT License.
