import anthropic
import os
import logging
from .prompt import PROMPT
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MaxRetriesExceededError(Exception):
    """Custom exception for when max retries are exceeded"""
    pass

class Claude:
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def emphasize(self, prompt, text: str):
        messages = []
        messages.append({"role": "user", "content": prompt.format(text=text)})
        
        max_retries = 5
        attempt = 0
        
        while attempt < max_retries:
            try:
                result = self.client.messages.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=4096,
                        tools = [
                            {
                                "name": "emphasize",
                                "description": "Emphasize selected words in the text",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {
                                        "emphasized_sentence": {
                                            "type": "string",
                                            "description": "The emphasized sentence"
                                        }
                                    }
                                }
                            }
                        ]
                )
                # Extract the emphasized_sentence from the tool calls
                for block in result.content:
                    if hasattr(block, 'type') and block.type == 'tool_use':
                        return block.input.get('emphasized_sentence')
                return None
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
                
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Final error: {str(e)}")
                raise MaxRetriesExceededError(f"Request failed after {max_retries} attempts. Last error: {str(e)}")

if __name__ == "__main__":
    claude = Claude()
    print(claude.emphasize(PROMPT, "Hello there, my name is David!"))