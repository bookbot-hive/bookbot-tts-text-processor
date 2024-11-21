import os
import logging
from openai import OpenAI
from .prompt import PROMPT
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MaxRetriesExceededError(Exception):
    """Custom exception for when max retries are exceeded"""
    pass

class GPT:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    def emphasize(self, prompt, text: str):
        messages = [{"role": "user", "content": prompt.format(text=text)}]
        
        max_retries = 5
        attempt = 0
        
        while attempt < max_retries:
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "emphasize",
                            "description": "Emphasize selected words in the text",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "emphasized_sentence": {
                                        "type": "string",
                                        "description": "The emphasized sentence"
                                    }
                                },
                                "required": ["emphasized_sentence"]
                            }
                        }
                    }],
                    tool_choice={"type": "function", "function": {"name": "emphasize"}}
                )
                
                # Extract the emphasized_sentence from the tool calls
                tool_calls = result.choices[0].message.tool_calls
                if tool_calls:
                    tool_call = tool_calls[0]
                    return json.loads(tool_call.function.arguments).get('emphasized_sentence')
                return None
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
            
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Final error: {str(e)}")
                raise MaxRetriesExceededError(f"Request failed after {max_retries} attempts. Last error: {str(e)}")

if __name__ == "__main__":
    gpt = GPT()
    print(gpt.emphasize(PROMPT, "Hello there, my name is David!"))
