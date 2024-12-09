import os
import logging
from openai import OpenAI
from pydantic import BaseModel

from .prompt import PROMPT
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MaxRetriesExceededError(Exception):
    """Custom exception for when max retries are exceeded"""
    pass

class EmphasizedSentence(BaseModel):
    emphasized_sentence: str

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
                result = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    response_format=EmphasizedSentence
                )
                result = result.choices[0].message.parsed
                return result.emphasized_sentence
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
            
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Final error: {str(e)}")
                raise MaxRetriesExceededError(f"Request failed after {max_retries} attempts. Last error: {str(e)}")

if __name__ == "__main__":
    gpt = GPT()
    print(gpt.emphasize(PROMPT, "Hello there, my name is David!"))
