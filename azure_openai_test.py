"""Azure OpenAI quick tester.

Runs three modes against your Azure OpenAI deployment:
- basic: single request
- multiturn: two-turn conversation (you manage history)
- stream: streamed output

Defaults to the newer Responses API because your Azure endpoint example uses:
  /openai/responses?api-version=2025-04-01-preview

Usage examples (PowerShell):
  $env:AZURE_OPENAI_API_KEY = "..."
  python azure_openai_test.py --mode basic
  python azure_openai_test.py --mode multiturn
  python azure_openai_test.py --mode stream
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import httpx
from openai import AzureOpenAI


def _try_load_dotenv() -> None:
	"""Load .env from the current working directory if python-dotenv is installed.

	This is optional. If python-dotenv isn't installed, we simply do nothing.
	"""

	env_path = Path(os.getcwd()) / ".env"
	if not env_path.exists():
		return

	try:
		from dotenv import load_dotenv  # type: ignore

		load_dotenv(dotenv_path=env_path)
	except Exception:
		# Keep this non-fatal: user can still set env vars manually.
		return


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
	value = os.getenv(name)
	if value is None or value == "":
		return default
	return value


def _require(value: Optional[str], hint: str) -> str:
	if not value:
		raise ValueError(hint)
	return value


def build_client(
	*,
	api_version: str,
	azure_endpoint: str,
	api_key: str,
	trust_env: bool,
) -> AzureOpenAI:
	# Some Windows setups set SSL_CERT_FILE/REQUESTS_CA_BUNDLE to a missing path,
	# which causes httpx to crash during client initialization.
	http_client = httpx.Client(trust_env=trust_env)
	return AzureOpenAI(
		api_version=api_version,
		azure_endpoint=azure_endpoint,
		api_key=api_key,
		http_client=http_client,
	)


def _print_debug(debug: bool, label: str, obj: Any) -> None:
	if not debug:
		return
	try:
		import json

		print(f"\n--- {label} (debug) ---")
		print(json.dumps(obj, indent=2, default=str))
		print("--- end ---\n")
	except Exception:
		print(f"\n--- {label} (debug, repr) ---")
		print(repr(obj))
		print("--- end ---\n")


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _prompt_question() -> str:
	# Single prompt read; supports multi-line input by pasting text.
	# End with Enter.
	try:
		return input("Question> ").strip()
	except EOFError:
		return ""


def _split_inline_context(raw: str) -> tuple[Optional[str], str]:
	"""Extract optional inline context from a user-provided string.

	Supported patterns (simple on purpose):
	- First line:  SYSTEM: <system/context text>
	  Remaining:   treated as the user question (optionally prefixed with USER:)

	- First paragraph: CONTEXT: <context text>
	  Blank line, then the user question.

	If no marker is found, returns (None, raw).
	"""

	text = (raw or "").strip()
	if not text:
		return None, ""

	lines = text.splitlines()
	first = lines[0].strip()
	upper_first = first.upper()
	if upper_first.startswith("SYSTEM:"):
		after = first.split(":", 1)[1].strip()
		if "||" in after:
			system_part, question_part = after.split("||", 1)
			return system_part.strip() or None, question_part.strip()
		system = after
		rest = "\n".join(lines[1:]).strip()
		if rest.upper().startswith("USER:"):
			rest = rest.split(":", 1)[1].strip()
		return system or None, rest

	if upper_first.startswith("CONTEXT:") or upper_first.startswith("CTX:"):
		label, ctx = first.split(":", 1)
		after = ctx.strip()
		if "||" in after:
			system_part, question_part = after.split("||", 1)
			return system_part.strip() or None, question_part.strip()
		system = after
		# Allow:
		# CONTEXT: ...
		# <blank>
		# question...
		rest_lines = lines[1:]
		while rest_lines and rest_lines[0].strip() == "":
			rest_lines = rest_lines[1:]
		rest = "\n".join(rest_lines).strip()
		if rest.upper().startswith("USER:"):
			rest = rest.split(":", 1)[1].strip()
		return system or None, rest

	return None, text


def _load_history(history_file: Path, default_system: str) -> tuple[str, List[Dict[str, str]]]:
	"""Load stored multi-turn history.

	We store ONLY user/assistant messages; system prompt is stored separately.
	"""
	if not history_file.exists():
		return default_system, []

	try:
		data = json.loads(history_file.read_text(encoding="utf-8"))
		system = str(data.get("system") or default_system)
		messages = data.get("messages") or []
		if not isinstance(messages, list):
			return system, []
		clean: List[Dict[str, str]] = []
		for item in messages:
			if not isinstance(item, dict):
				continue
			role = item.get("role")
			content = item.get("content")
			if role in ("user", "assistant") and isinstance(content, str):
				clean.append({"role": role, "content": content})
		return system, clean
	except Exception:
		return default_system, []


def _trim_history(messages: List[Dict[str, str]], keep_pairs: int) -> List[Dict[str, str]]:
	"""Keep only the last N (user, assistant) pairs."""
	if keep_pairs <= 0:
		return []
	# Expect alternating user/assistant; be robust anyway.
	max_messages = keep_pairs * 2
	trimmed = [m for m in messages if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
	return trimmed[-max_messages:]


def _save_history(history_file: Path, system: str, messages: List[Dict[str, str]]) -> None:
	data = {"system": system, "messages": messages}
	history_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_text_file(path: Path) -> str:
	try:
		if not path.exists():
			return ""
		return path.read_text(encoding="utf-8").strip()
	except Exception:
		return ""


def _save_text_file(path: Path, text: str) -> None:
	path.write_text((text or "").strip() + "\n", encoding="utf-8")


def _load_stream_history(path: Path) -> List[Dict[str, str]]:
	"""Load stream history as a flat list of {role, content} for user/assistant."""
	if not path.exists():
		return []
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		messages = data.get("messages") if isinstance(data, dict) else data
		if not isinstance(messages, list):
			return []
		clean: List[Dict[str, str]] = []
		for item in messages:
			if not isinstance(item, dict):
				continue
			role = item.get("role")
			content = item.get("content")
			if role in ("user", "assistant") and isinstance(content, str):
				clean.append({"role": role, "content": content})
		return clean
	except Exception:
		return []


def _save_stream_history(path: Path, messages: List[Dict[str, str]]) -> None:
	data = {"messages": messages}
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_stream_memory(
	client: AzureOpenAI,
	api: str,
	deployment: str,
	*,
	seed_context: str,
	pairs: List[Dict[str, str]],
	max_output_tokens: int,
	debug: bool,
) -> str:
	"""Create a compact 'memory' string from the last few Q/A pairs.

	This runs an extra model call after each streamed answer. The result is saved
	to a local file and used as additional system context in subsequent stream runs.
	"""

	# Format pairs as readable transcript.
	transcript_lines: List[str] = []
	for m in pairs:
		role = m.get("role")
		content = (m.get("content") or "").strip()
		if not content:
			continue
		if role == "user":
			transcript_lines.append(f"User: {content}")
		elif role == "assistant":
			transcript_lines.append(f"Assistant: {content}")

	transcript = "\n".join(transcript_lines).strip()

	messages = [
		{
			"role": "system",
			"content": (
				"You are a tool that writes short conversation memory for a chat assistant. "
				"Given the recent user questions and assistant answers, produce a compact MEMORY "
				"that will be appended to the system prompt for the next request. "
				"Rules: keep it under 8 bullet points; include stable user preferences/constraints; "
				"include important facts the user stated; do NOT include the assistant's internal reasoning; "
				"avoid sensitive personal data; be concise and concrete. "
				"Output ONLY the bullet list (no intro text)."
			),
		},
		{
			"role": "user",
			"content": (
				f"Seed context (style/instructions):\n{seed_context.strip() or '(none)'}\n\n"
				f"Recent transcript:\n{transcript or '(empty)'}"
			),
		},
	]

	# Use a smaller token cap for memory generation.
	text = _call_text_auto(
		client,
		api,
		deployment,
		messages,
		max_output_tokens,
		debug,
	)
	return text.strip()


def _build_messages(
	*,
	system_prompt: str,
	inline_system: Optional[str],
	history_messages: List[Dict[str, str]],
	user_question: str,
) -> List[Dict[str, str]]:
	final_system = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
	if inline_system:
		# Treat inline context as part of the system prompt.
		final_system = f"{final_system}\n\nContext: {inline_system.strip()}"

	messages: List[Dict[str, str]] = [{"role": "system", "content": final_system}]
	messages.extend(history_messages)
	messages.append({"role": "user", "content": user_question})
	return messages


def _responses_text(
	client: AzureOpenAI,
	deployment: str,
	messages: List[Dict[str, str]],
	max_output_tokens: int,
	debug: bool,
) -> str:
	resp = client.responses.create(
		model=deployment,
		input=messages,
		max_output_tokens=max_output_tokens,
	)
	_print_debug(debug, "responses.text", resp)
	return str(getattr(resp, "output_text", "") or "")


def _responses_stream(
	client: AzureOpenAI,
	deployment: str,
	messages: List[Dict[str, str]],
	max_output_tokens: int,
	debug: bool,
) -> str:
	stream = client.responses.create(
		model=deployment,
		input=messages,
		max_output_tokens=max_output_tokens,
		stream=True,
	)

	collected: List[str] = []
	for event in stream:
		etype = getattr(event, "type", None)
		if etype == "response.output_text.delta":
			delta = getattr(event, "delta", "") or ""
			if delta:
				collected.append(delta)
				print(delta, end="", flush=True)
		elif debug:
			_print_debug(True, "responses.stream.event", event)

	print("")
	return "".join(collected).strip()


def _chat_text(
	client: AzureOpenAI,
	deployment: str,
	messages: List[Dict[str, str]],
	max_completion_tokens: int,
	debug: bool,
) -> str:
	resp = client.chat.completions.create(
		model=deployment,
		messages=messages,
		max_completion_tokens=max_completion_tokens,
	)
	_print_debug(debug, "chat.text", resp)
	return str(resp.choices[0].message.content or "")


def _chat_stream(
	client: AzureOpenAI,
	deployment: str,
	messages: List[Dict[str, str]],
	max_completion_tokens: int,
	debug: bool,
) -> str:
	stream = client.chat.completions.create(
		model=deployment,
		stream=True,
		messages=messages,
		max_completion_tokens=max_completion_tokens,
	)

	collected: List[str] = []
	for update in stream:
		if not getattr(update, "choices", None):
			continue
		delta = update.choices[0].delta
		text = getattr(delta, "content", None) or ""
		if text:
			collected.append(text)
			print(text, end="", flush=True)
		elif debug:
			_print_debug(True, "chat.stream.update", update)

	print("")
	return "".join(collected).strip()


def _call_text_auto(
	client: AzureOpenAI,
	api: str,
	deployment: str,
	messages: List[Dict[str, str]],
	max_output_tokens: int,
	debug: bool,
) -> str:
	if api == "responses":
		return _responses_text(client, deployment, messages, max_output_tokens, debug)
	if api == "chat_completions":
		return _chat_text(client, deployment, messages, max_output_tokens, debug)

	# auto
	try:
		return _responses_text(client, deployment, messages, max_output_tokens, debug)
	except Exception as exc:
		print("Responses API failed; falling back to chat.completions...", file=sys.stderr)
		if debug:
			print(f"Responses error: {exc}", file=sys.stderr)
		return _chat_text(client, deployment, messages, max_output_tokens, debug)


def _call_stream_auto(
	client: AzureOpenAI,
	api: str,
	deployment: str,
	messages: List[Dict[str, str]],
	max_output_tokens: int,
	debug: bool,
) -> str:
	if api == "responses":
		return _responses_stream(client, deployment, messages, max_output_tokens, debug)
	if api == "chat_completions":
		return _chat_stream(client, deployment, messages, max_output_tokens, debug)

	# auto
	try:
		return _responses_stream(client, deployment, messages, max_output_tokens, debug)
	except Exception as exc:
		print("Responses API failed; falling back to chat.completions...", file=sys.stderr)
		if debug:
			print(f"Responses error: {exc}", file=sys.stderr)
		return _chat_stream(client, deployment, messages, max_output_tokens, debug)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Test Azure OpenAI deployment (basic/multiturn/stream)."
	)
	parser.add_argument("--mode", choices=["basic", "multiturn", "stream"], default="basic")
	parser.add_argument(
		"--api",
		choices=["responses", "chat_completions", "auto"],
		default="auto",
		help="Which API surface to use. 'auto' tries Responses then falls back to Chat Completions.",
	)

	parser.add_argument(
		"--question",
		"-q",
		default=None,
		help=(
			"Your question text. If omitted, the script will prompt. "
			"You can include inline context like: 'SYSTEM: <context>\\nUSER: <question>'."
		),
	)
	parser.add_argument(
		"--system",
		default=_env("AZURE_OPENAI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
		help="System prompt used for all requests (inline SYSTEM: overrides/extends this).",
	)

	parser.add_argument(
		"--endpoint",
		default=_env(
			"AZURE_OPENAI_ENDPOINT",
			"https://marko-mkhcc3vv-eastus2.cognitiveservices.azure.com/",
		),
		help="Azure resource endpoint base URL (no /openai/... path).",
	)
	parser.add_argument(
		"--deployment",
		default=_env("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat"),
		help="Deployment name in Azure OpenAI Studio.",
	)
	parser.add_argument(
		"--api-version",
		default=_env("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
		help="Azure OpenAI API version.",
	)
	parser.add_argument(
		"--max-output-tokens",
		type=int,
		default=int(_env("AZURE_OPENAI_MAX_OUTPUT_TOKENS", _env("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "512")) or "512"),
		help="Token limit for output. Start small while testing.",
	)
	parser.add_argument(
		"--history-file",
		default=_env("AZURE_OPENAI_HISTORY_FILE", str(Path(os.getcwd()) / ".azure_openai_history.json")),
		help="Where to store multi-turn history (JSON). Used only in --mode multiturn.",
	)
	parser.add_argument(
		"--history-turns",
		type=int,
		default=int(_env("AZURE_OPENAI_HISTORY_TURNS", "2") or "2"),
		help="How many (question, answer) pairs to keep in history for multiturn.",
	)
	parser.add_argument(
		"--reset-history",
		action="store_true",
		help="Clear stored multi-turn history before running (multiturn mode only).",
	)
	parser.add_argument(
		"--stream-history-file",
		default=_env(
			"AZURE_OPENAI_STREAM_HISTORY_FILE",
			str(Path(os.getcwd()) / ".azure_openai_stream_history.json"),
		),
		help="Where to store stream Q/A history (JSON). Used only in --mode stream.",
	)
	parser.add_argument(
		"--stream-context-file",
		default=_env(
			"AZURE_OPENAI_STREAM_CONTEXT_FILE",
			str(Path(os.getcwd()) / ".azure_openai_stream_context.txt"),
		),
		help="Where to store the auto-built stream context (text). Used only in --mode stream.",
	)
	parser.add_argument(
		"--stream-context-turns",
		type=int,
		default=int(_env("AZURE_OPENAI_STREAM_CONTEXT_TURNS", "2") or "2"),
		help="How many (Q,A) pairs to use when building stream context.",
	)
	parser.add_argument(
		"--stream-context-max-tokens",
		type=int,
		default=int(_env("AZURE_OPENAI_STREAM_CONTEXT_MAX_TOKENS", "256") or "256"),
		help="Token cap for the extra request that builds the updated stream context.",
	)
	parser.add_argument(
		"--api-key",
		default=_env("AZURE_OPENAI_API_KEY"),
		help="API key (prefer AZURE_OPENAI_API_KEY env var).",
	)
	parser.add_argument(
		"--trust-env",
		action="store_true",
		help="Let httpx read proxy/SSL env vars (may break if SSL_CERT_FILE is invalid).",
	)
	parser.add_argument("--debug", action="store_true", help="Print raw responses/events.")
	return parser.parse_args()


def _resolve_question(args: argparse.Namespace) -> str:
	question = (args.question or "").strip()
	if question:
		return question
	return _prompt_question()


def run_basic(client: AzureOpenAI, args: argparse.Namespace, deployment: str) -> None:
	raw = _resolve_question(args)
	inline_system, question = _split_inline_context(raw)
	if not question:
		raise ValueError("Empty question.")

	messages = _build_messages(
		system_prompt=args.system,
		inline_system=inline_system,
		history_messages=[],
		user_question=question,
	)
	answer = _call_text_auto(client, args.api, deployment, messages, args.max_output_tokens, args.debug)
	print(answer)


def run_stream(client: AzureOpenAI, args: argparse.Namespace, deployment: str) -> None:
	# Same as basic, but streamed. Inline context is especially useful here.
	raw = _resolve_question(args)
	inline_system, question = _split_inline_context(raw)
	if not question:
		raise ValueError("Empty question.")

	# Load dynamic stream context (built from previous streamed Q/A pairs).
	stream_history_file = Path(args.stream_history_file)
	stream_context_file = Path(args.stream_context_file)
	stream_history = _load_stream_history(stream_history_file)
	auto_context = _load_text_file(stream_context_file)

	combined_inline: Optional[str]
	if inline_system and auto_context:
		combined_inline = f"{inline_system.strip()}\n\nMemory (auto, last {int(args.stream_context_turns)} turns):\n{auto_context.strip()}"
	elif inline_system:
		combined_inline = inline_system
	elif auto_context:
		combined_inline = f"Memory (auto, last {int(args.stream_context_turns)} turns):\n{auto_context.strip()}"
	else:
		combined_inline = None

	messages = _build_messages(
		system_prompt=args.system,
		inline_system=combined_inline,
		history_messages=[],
		user_question=question,
	)
	answer = _call_stream_auto(client, args.api, deployment, messages, args.max_output_tokens, args.debug)

	# Update stream history (kept locally; not sent directly as messages in stream mode).
	updated = stream_history + [
		{"role": "user", "content": question},
		{"role": "assistant", "content": answer},
	]
	updated = _trim_history(updated, keep_pairs=int(args.stream_context_turns))
	_save_stream_history(stream_history_file, updated)

	# Build new auto-context from the last N pairs and save it. This does not touch azure_openai_questions.json.
	seed = (inline_system or "").strip()
	new_context = _build_stream_memory(
		client,
		args.api,
		deployment,
		seed_context=seed,
		pairs=updated,
		max_output_tokens=int(args.stream_context_max_tokens),
		debug=bool(args.debug),
	)
	if new_context:
		_save_text_file(stream_context_file, new_context)


def run_multiturn(client: AzureOpenAI, args: argparse.Namespace, deployment: str) -> None:
	history_file = Path(args.history_file)
	if args.reset_history and history_file.exists():
		history_file.unlink()

	system_from_file, history_messages = _load_history(history_file, default_system=args.system)
	# Command-line --system always wins as baseline, but keep any stored system prompt if the user
	# didn't override it.
	base_system = args.system.strip() if (args.system or "").strip() else system_from_file

	raw = _resolve_question(args)
	inline_system, question = _split_inline_context(raw)
	if not question:
		raise ValueError("Empty question.")

	messages = _build_messages(
		system_prompt=base_system,
		inline_system=inline_system,
		history_messages=history_messages,
		user_question=question,
	)
	answer = _call_text_auto(client, args.api, deployment, messages, args.max_output_tokens, args.debug)
	print(answer)

	# Persist only user/assistant messages, trimmed to last N pairs.
	new_history = history_messages + [
		{"role": "user", "content": question},
		{"role": "assistant", "content": answer},
	]
	new_history = _trim_history(new_history, keep_pairs=int(args.history_turns))
	_save_history(history_file, base_system, new_history)


def main() -> int:
	_try_load_dotenv()
	args = parse_args()
	try:
		api_key = _require(args.api_key, "Missing API key. Set AZURE_OPENAI_API_KEY or pass --api-key.")
		endpoint = _require(args.endpoint, "Missing endpoint. Set AZURE_OPENAI_ENDPOINT or pass --endpoint.")
		deployment = _require(args.deployment, "Missing deployment. Set AZURE_OPENAI_DEPLOYMENT or pass --deployment.")
	except ValueError as e:
		print(str(e), file=sys.stderr)
		return 2

	client = build_client(
		api_version=args.api_version,
		azure_endpoint=endpoint,
		api_key=api_key,
		trust_env=bool(args.trust_env),
	)
	try:
		if args.mode == "basic":
			run_basic(client, args, deployment)
		elif args.mode == "stream":
			run_stream(client, args, deployment)
		elif args.mode == "multiturn":
			run_multiturn(client, args, deployment)
		else:
			raise ValueError(f"Unknown mode: {args.mode}")
	except Exception as exc:
		print("Request failed.", file=sys.stderr)
		print(f"Endpoint: {endpoint}", file=sys.stderr)
		print(f"Deployment: {deployment}", file=sys.stderr)
		print(f"API version: {args.api_version}", file=sys.stderr)
		print(f"Error: {exc}", file=sys.stderr)
		return 1
	finally:
		client.close()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())