"""
Claude Code CLI wrapper for executing prompts using Claude Opus 4.5
Based on the working implementation from prompt_creator.py
"""

import subprocess
import json
import os
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MaxRetriesExceededError(Exception):
    """Custom exception for when max retries are exceeded"""

    pass


EMPHASIZE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "emphasized_sentence": {
            "type": "string",
            "description": "The emphasized sentence",
        }
    },
    "required": ["emphasized_sentence"],
    "additionalProperties": False,
}


def execute_prompt(
    prompt,
    output_dir="output/claude_responses",
    timeout=120,
    max_retries=5,
    max_backoff_time=3600,
    model="opus",
    output_json=False,
    json_schema=None,
):
    """Execute a prompt using Claude Code CLI and return the result.
    
    This uses the Claude CLI with the specified model via stdin.
    Uses 'opus' for Claude Opus 4.5 (most capable model).
    
    Args:
        prompt (str): The prompt to send to Claude Code CLI
        output_dir (str): Directory to save response JSON files
        timeout (int): Timeout in seconds (default: 120)
        max_retries (int): Maximum number of retry attempts for rate limits (default: 5)
        max_backoff_time (int): Maximum total backoff time in seconds (default: 3600 = 1 hour)
        model (str): Model to use - 'opus', 'sonnet', etc. (default: 'opus')
        output_json (bool): If True, write response to JSON file; if False, return raw response only (default: False)
        json_schema (dict): Optional JSON schema for Claude CLI structured output
        
    Returns:
        dict: Response containing 'success', 'response', and optionally 'filepath' and metadata
    """
    try:
        # print("=" * 80)
        # print("CLAUDE CODE CLI PROMPT EXECUTION")
        # print("=" * 80)
        # print(f"Prompt length: {len(prompt)} chars")
        # print(f"Timeout: {timeout}s")
        # print(f"Max retries: {max_retries}, Max backoff: {max_backoff_time}s")
        # print(f"Model: {model}")
        # print("-" * 80)
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get Claude CLI path (configurable via env)
        claude_path = os.getenv('CLAUDE_CLI_PATH', 'claude')
        
        # Check if CLI is available
        if not shutil.which(claude_path):
            error_msg = (
                f"Claude CLI not found at '{claude_path}'. "
                f"Install with: npm install -g @anthropics/claude-cli"
            )
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        
        # Build command using print mode so stdin returns non-interactive output.
        # Use 'opus' for Claude Opus 4.5
        # -p (--print) outputs response only, no interactive mode
        cmd = [claude_path, '-p', '--model', model]
        if json_schema:
            cmd.extend(['--json-schema', json.dumps(json_schema)])
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Retry loop with exponential backoff
        total_backoff = 0
        result = None
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries - 1}")
            
            # Execute Claude CLI with the prompt on stdin.
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            # Success - process response
            if result.returncode == 0:
                break
            
            # Check if this is a rate limit error
            is_rate_limit = False
            combined_output = f"{result.stdout}\n{result.stderr}"
            if combined_output:
                stdout_lower = combined_output.lower()
                is_rate_limit = any([
                    'rate_limit' in stdout_lower,
                    'rate limit' in stdout_lower,
                    'overloaded' in stdout_lower,
                    'too many requests' in stdout_lower,
                    '429' in combined_output
                ])
            
            # If not a rate limit error or last attempt, fail immediately
            if not is_rate_limit or attempt == max_retries - 1:
                error_msg = f"Claude CLI failed with exit code {result.returncode}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"STDOUT: {result.stdout}"
                print(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'exit_code': result.returncode,
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }
            
            # Calculate backoff time (exponential: 60s, 120s, 240s, 480s, 960s)
            backoff_time = min(60 * (2 ** attempt), max_backoff_time - total_backoff)
            
            # Check if we've exceeded max backoff time
            if total_backoff + backoff_time > max_backoff_time:
                remaining_time = max_backoff_time - total_backoff
                if remaining_time > 0:
                    backoff_time = remaining_time
                else:
                    error_msg = f"Max backoff time ({max_backoff_time}s) exceeded. Last error:\n"
                    error_msg += f"STDOUT: {result.stdout}"
                    print(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'stdout': result.stdout
                    }
            
            print(f"⚠️  Rate limit detected. Retrying in {backoff_time}s...")
            print(f"   Total backoff so far: {total_backoff}s / {max_backoff_time}s")
            
            time.sleep(backoff_time)
            total_backoff += backoff_time
        
        # Log final retry stats if any retries occurred
        if total_backoff > 0:
            print(f"✅ Request succeeded after {total_backoff}s of backoff")
        
        # Extract response text from stdout
        response_text = result.stdout.strip()
        
        # If output_json is True, save to JSON file
        if output_json:
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"claude_cli_response_{timestamp}.json"
            json_filepath = Path(output_dir) / json_filename
            
            # Save raw response to JSON file
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'provider': 'claude_cli',
                'model': model,
                'prompt_length': len(prompt),
                'response_length': len(response_text),
                'raw_output': response_text,
                'stderr': result.stderr,
                'exit_code': result.returncode,
                'total_backoff_time': total_backoff,
                'json_schema': json_schema,
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            
            print("-" * 80)
            print(f"✅ Success! Response saved to: {json_filepath}")
            print(f"Response length: {len(response_text)} chars")
            print("=" * 80)
            
            return {
                'success': True,
                'response': response_text,
                'filepath': str(json_filepath),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model': model,
                    'prompt_length': len(prompt),
                    'response_length': len(response_text),
                    'total_backoff_time': total_backoff
                }
            }
        else:
            # Return only raw response text
            print("-" * 80)
            print(f"✅ Success! Response length: {len(response_text)} chars")
            print("=" * 80)
            
            return {
                'success': True,
                'response': response_text
            }
        
    except subprocess.TimeoutExpired:
        error_msg = f'Claude CLI execution timed out after {timeout} seconds'
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_msg = f'Unexpected error: {str(e)}'
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'exception': str(e)
        }  

def _parse_json_response(response_text):
    """Parse JSON from direct CLI output or a markdown fenced JSON block."""
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("Claude CLI returned an empty response")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fenced_content = "\n".join(lines).strip()
        try:
            return json.loads(fenced_content)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from Claude CLI response: {response_text}")


def _extract_emphasized_sentence(response_text):
    parsed = _parse_json_response(response_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    emphasized_sentence = parsed.get("emphasized_sentence")
    if not isinstance(emphasized_sentence, str):
        raise ValueError("Claude CLI response is missing string field 'emphasized_sentence'")

    return emphasized_sentence


class ClaudeCodeCLI:
    """Drop-in emphasis client that uses the authenticated Claude Code CLI."""

    def __init__(
        self,
        model: str = "opus",
        timeout: int = 120,
        output_dir: str = "output/claude_responses",
        max_retries: int = 5,
        max_backoff_time: int = 3600,
        output_json: bool = False,
    ):
        self.model = model
        self.timeout = timeout
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_backoff_time = max_backoff_time
        self.output_json = output_json

    def emphasize(self, prompt, text: str):
        formatted_prompt = prompt.format(text=text)
        structured_prompt = (
            f"{formatted_prompt}\n\n"
            "Return only valid JSON matching this exact shape, with no markdown "
            "fences and no explanation:\n"
            '{"emphasized_sentence": "the full sentence with selected words in square brackets"}'
        )

        result = execute_prompt(
            prompt=structured_prompt,
            output_dir=self.output_dir,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_backoff_time=self.max_backoff_time,
            model=self.model,
            output_json=self.output_json,
            json_schema=EMPHASIZE_RESPONSE_SCHEMA,
        )

        if not result.get("success"):
            error = result.get("error", "Claude CLI request failed")
            logger.error(error)
            raise MaxRetriesExceededError(error)

        try:
            return _extract_emphasized_sentence(result["response"])
        except Exception as e:
            logger.error("Failed to parse Claude CLI emphasis response", exc_info=True)
            raise MaxRetriesExceededError(str(e))


Claude = ClaudeCodeCLI


def emphasize(
    prompt,
    text: str,
    model: str = "opus",
    timeout: int = 120,
    output_dir: str = "output/claude_responses",
    max_retries: int = 5,
    max_backoff_time: int = 3600,
    output_json: bool = False,
):
    return ClaudeCodeCLI(
        model=model,
        timeout=timeout,
        output_dir=output_dir,
        max_retries=max_retries,
        max_backoff_time=max_backoff_time,
        output_json=output_json,
    ).emphasize(prompt, text)


def main():
    try:
        from .prompt import PROMPT
    except ImportError:
        from prompt import PROMPT

    claude = ClaudeCodeCLI()
    print(claude.emphasize(PROMPT, "Hello there, my name is David!"))

if __name__ == "__main__":
    main()
