"""
Codex SDK wrapper for executing prompts with codex-python-sdk.

This mirrors claude_code_cli.execute_prompt() enough for existing call sites:
execute_prompt(...) returns a dict with success/response on success, or
success/error on failure. It uses the local Codex CLI app-server through the
`codex-python-sdk` package.

Run this file directly for a small sample:
    python codex_cli.py
"""

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from codex_python_sdk import (
    AppServerConnectionError,
    CodexAgenticError,
    NotAuthenticatedError,
    create_client,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_MODEL = "gpt-5.5"


class MaxRetriesExceededError(Exception):
    """Custom exception for when max retries are exceeded"""

    pass


def _is_retryable_error(exc):
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return any(
        marker in text or marker in name
        for marker in (
            "rate_limit",
            "rate limit",
            "too many requests",
            "overloaded",
            "429",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection",
        )
    )


def _build_thread_params(model=None, approval_policy="never", sandbox="read-only"):
    params = {}
    if model:
        params["model"] = model
    if approval_policy:
        params["approvalPolicy"] = approval_policy
    if sandbox:
        params["sandbox"] = sandbox
    return params


def execute_prompt(
    prompt,
    output_dir="output/codex_responses",
    timeout=120,
    max_retries=5,
    max_backoff_time=3600,
    model=DEFAULT_MODEL,
    output_json=False,
    approval_policy="never",
    sandbox="read-only",
    cwd=None,
    include_events=False,
):
    """Execute a prompt using Codex and return a result dict.

    Args:
        prompt (str): Prompt to send to Codex.
        output_dir (str): Directory for optional JSON response files.
        timeout (int): App-server stream idle timeout in seconds.
        max_retries (int): Maximum attempts for retryable errors.
        max_backoff_time (int): Maximum total retry sleep in seconds.
        model (str | None): Optional Codex model name. Defaults to gpt-5.5.
        output_json (bool): Save response metadata to JSON when True.
        approval_policy (str): Codex approval policy. Defaults to "never".
        sandbox (str): Codex sandbox. Defaults to "read-only".
        cwd (str | None): Working directory for Codex app-server.
        include_events (bool): Include SDK event objects in metadata when True.

    Returns:
        dict: {success, response, ...} on success or {success, error, ...}.
    """
    if not prompt:
        return {"success": False, "error": "Prompt is empty"}

    codex_command = os.getenv("CODEX_CLI_PATH") or os.getenv("CODEX_COMMAND") or "codex"
    if not shutil.which(codex_command):
        error_msg = (
            f"Codex CLI not found at '{codex_command}'. "
            "Install/login Codex first, or set CODEX_CLI_PATH."
        )
        print(f"Error: {error_msg}")
        return {"success": False, "error": error_msg}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    process_cwd = cwd or os.getcwd()
    active_model = model or DEFAULT_MODEL
    thread_params = _build_thread_params(
        model=active_model,
        approval_policy=approval_policy,
        sandbox=sandbox,
    )

    total_backoff = 0
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"Retry attempt {attempt}/{max_retries - 1}")

        try:
            with create_client(
                codex_command=codex_command,
                process_cwd=process_cwd,
                default_thread_params=thread_params,
                stream_idle_timeout_seconds=timeout,
                on_command_approval=lambda params: {"decision": "decline"},
                on_file_change_approval=lambda params: {"decision": "decline"},
                on_permissions_approval=lambda params: {
                    "permissions": {},
                    "scope": "turn",
                },
                on_tool_request_user_input=lambda params: {"answers": {}},
            ) as client:
                response = client.responses_create(
                    prompt=prompt,
                    include_events=include_events,
                )

            response_text = response.text.strip()
            if output_json:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filepath = Path(output_dir) / f"codex_response_{timestamp}.json"
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "provider": "codex_python_sdk",
                    "model": active_model,
                    "prompt_length": len(prompt),
                    "response_length": len(response_text),
                    "raw_output": response_text,
                    "thread_id": response.thread_id,
                    "request_id": response.request_id,
                    "tool_name": response.tool_name,
                    "total_backoff_time": total_backoff,
                }
                if include_events and response.events is not None:
                    response_data["events"] = [
                        {
                            "type": event.type,
                            "phase": event.phase,
                            "text_delta": event.text_delta,
                            "message_text": event.message_text,
                            "request_id": event.request_id,
                            "thread_id": event.thread_id,
                            "turn_id": event.turn_id,
                            "timestamp": event.timestamp,
                        }
                        for event in response.events
                    ]
                with open(json_filepath, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, indent=2, ensure_ascii=False)

                print("-" * 80)
                print(f"Success! Response saved to: {json_filepath}")
                print(f"Response length: {len(response_text)} chars")
                print("=" * 80)
                return {
                    "success": True,
                    "response": response_text,
                    "filepath": str(json_filepath),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "model": active_model,
                        "prompt_length": len(prompt),
                        "response_length": len(response_text),
                        "total_backoff_time": total_backoff,
                    },
                }

            print("-" * 80)
            print(f"Success! Response length: {len(response_text)} chars")
            print("=" * 80)
            return {
                "success": True,
                "response": response_text,
            }

        except NotAuthenticatedError as e:
            error_msg = (
                "Codex is not authenticated. Run `codex login` or set "
                "CODEX_API_KEY."
            )
            print(f"Error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "exception": str(e),
            }
        except (AppServerConnectionError, CodexAgenticError, Exception) as e:
            last_error = e
            if not _is_retryable_error(e) or attempt == max_retries - 1:
                error_msg = f"Codex execution failed: {type(e).__name__}: {e}"
                print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "total_backoff_time": total_backoff,
                }

            backoff_time = min(60 * (2 ** attempt), max_backoff_time - total_backoff)
            if backoff_time <= 0:
                break
            print(f"Retryable Codex error. Retrying in {backoff_time}s...")
            print(f"   Error: {type(e).__name__}: {e}")
            time.sleep(backoff_time)
            total_backoff += backoff_time

    error_msg = f"Max backoff time exceeded. Last error: {last_error}"
    print(f"Error: {error_msg}")
    return {
        "success": False,
        "error": error_msg,
        "total_backoff_time": total_backoff,
    }


def _parse_json_response(response_text):
    """Parse JSON from direct Codex output or a markdown fenced JSON block."""
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("Codex returned an empty response")

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

    raise ValueError(f"Could not parse JSON from Codex response: {response_text}")


def _extract_emphasized_sentence(response_text):
    parsed = _parse_json_response(response_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    emphasized_sentence = parsed.get("emphasized_sentence")
    if not isinstance(emphasized_sentence, str):
        raise ValueError("Codex response is missing string field 'emphasized_sentence'")

    return emphasized_sentence


class CodexCLI:
    """Drop-in emphasis client that uses the authenticated Codex CLI."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
        output_dir: str = "output/codex_responses",
        max_retries: int = 5,
        max_backoff_time: int = 3600,
        output_json: bool = False,
        approval_policy: str = "never",
        sandbox: str = "read-only",
        cwd: str = None,
    ):
        self.model = model
        self.timeout = timeout
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_backoff_time = max_backoff_time
        self.output_json = output_json
        self.approval_policy = approval_policy
        self.sandbox = sandbox
        self.cwd = cwd

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
            approval_policy=self.approval_policy,
            sandbox=self.sandbox,
            cwd=self.cwd,
        )

        if not result.get("success"):
            error = result.get("error", "Codex request failed")
            logger.error(error)
            raise MaxRetriesExceededError(error)

        try:
            return _extract_emphasized_sentence(result["response"])
        except Exception as e:
            logger.error("Failed to parse Codex emphasis response", exc_info=True)
            raise MaxRetriesExceededError(str(e))


def emphasize(
    prompt,
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    output_dir: str = "output/codex_responses",
    max_retries: int = 5,
    max_backoff_time: int = 3600,
    output_json: bool = False,
    approval_policy: str = "never",
    sandbox: str = "read-only",
    cwd: str = None,
):
    return CodexCLI(
        model=model,
        timeout=timeout,
        output_dir=output_dir,
        max_retries=max_retries,
        max_backoff_time=max_backoff_time,
        output_json=output_json,
        approval_policy=approval_policy,
        sandbox=sandbox,
        cwd=cwd,
    ).emphasize(prompt, text)


Codex = CodexCLI


def main():
    try:
        from .prompt import PROMPT
    except ImportError:
        from prompt import PROMPT

    codex = CodexCLI()
    print(codex.emphasize(PROMPT, "Hello there, my name is David!"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
