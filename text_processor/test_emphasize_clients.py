"""Smoke-test emphasize() for Claude, GPT, ClaudeCodeCLI, and CodexCLI."""

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path


SAMPLE_TEXT = (
    "I didn't say she stole the red bike; I said she borrowed it yesterday."
)


@dataclass
class ClientSpec:
    label: str
    module_name: str
    class_name: str
    model: str


CLIENTS = [
    ClientSpec("Claude", "claude", "Claude", "claude-3-5-sonnet-20240620"),
    ClientSpec("GPT", "gpt", "GPT", "gpt-4o"),
    ClientSpec("ClaudeCodeCLI", "claude_code_cli", "ClaudeCodeCLI", "sonnet"),
    ClientSpec("CodexCLI", "codex_cli", "CodexCLI", "gpt-5.5"),
]


def _ensure_local_package():
    package_dir = Path(__file__).resolve().parent
    package_name = package_dir.name

    if str(package_dir.parent) not in sys.path:
        sys.path.insert(0, str(package_dir.parent))

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    return package_name


def _import_from_local_package(module_name):
    package_name = _ensure_local_package()
    return importlib.import_module(f"{package_name}.{module_name}")


def _build_client(spec):
    module = _import_from_local_package(spec.module_name)
    client_class = getattr(module, spec.class_name)
    return client_class(model=spec.model)


def run_client(spec, prompt):
    print(f"\n[{spec.label}] model={spec.model}")
    try:
        client = _build_client(spec)
        result = client.emphasize(prompt, SAMPLE_TEXT)
        print(f"OK: {result}")
        return True
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return False


def main():
    prompt_module = _import_from_local_package("prompt")
    prompt = prompt_module.PROMPT

    print(f"Sample text: {SAMPLE_TEXT}")

    failures = 0
    for spec in CLIENTS:
        ok = run_client(spec, prompt)
        if not ok:
            failures += 1

    print(f"\nDone: {len(CLIENTS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
