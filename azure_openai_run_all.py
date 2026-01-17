"""Run all Azure OpenAI test modes in one go.

This script calls azure_openai_test.py as a subprocess so it exercises the same
CLI paths you use manually.

Prereqs:
- .env must contain AZURE_OPENAI_API_KEY (+ endpoint/deployment/api version)
- Packages: openai, python-dotenv

Run:
  python azure_openai_run_all.py
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


def _try_load_dotenv() -> None:
	env_path = Path(os.getcwd()) / ".env"
	if not env_path.exists():
		return
	try:
		from dotenv import load_dotenv  # type: ignore

		load_dotenv(dotenv_path=env_path)
	except Exception:
		return


def _require_env(name: str) -> str:
	value = os.getenv(name)
	if not value:
		raise SystemExit(f"Missing {name}. Put it in .env or set it in your environment.")
	return value


def run_cmd(args: list[str]) -> None:
	print(f"\n$ {' '.join(args)}")
	subprocess.run(args, check=True)


def main() -> int:
	_try_load_dotenv()

	_require_env("AZURE_OPENAI_API_KEY")
	_require_env("AZURE_OPENAI_ENDPOINT")
	_require_env("AZURE_OPENAI_DEPLOYMENT")
	_require_env("AZURE_OPENAI_API_VERSION")

	python = sys.executable
	script = str(Path(__file__).with_name("azure_openai_test.py"))

	questions_path = Path(__file__).with_name("azure_openai_questions.json")
	if not questions_path.exists():
		raise SystemExit(
			f"Missing {questions_path.name}. Create it (see example in repo root)."
		)

	try:
		cfg = json.loads(questions_path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise SystemExit(f"Failed to read {questions_path.name}: {exc}")

	basic_q = str((cfg.get("basic") or {}).get("question") or "").strip()
	multiturn_qs = (cfg.get("multiturn") or {}).get("questions") or []
	if not isinstance(multiturn_qs, list):
		multiturn_qs = []
	multiturn_qs = [str(q).strip() for q in multiturn_qs if str(q).strip()]

	stream_ctx = str((cfg.get("stream") or {}).get("context") or "").strip()
	stream_q = str((cfg.get("stream") or {}).get("question") or "").strip()

	history_file = str(Path(os.getcwd()) / ".azure_openai_history.json")

	# 1) basic
	if basic_q:
		run_cmd([python, script, "--mode", "basic", "-q", basic_q])
	else:
		print("\n[skip] basic.question is empty in azure_openai_questions.json")

	# 2) multiturn - keep only last 2 (Q,A) pairs; runner can run any number of questions.
	if multiturn_qs:
		# Reset once at the start.
		run_cmd(
			[
				python,
				script,
				"--mode",
				"multiturn",
				"--reset-history",
				"--history-file",
				history_file,
				"--history-turns",
				"2",
				"-q",
				multiturn_qs[0],
			]
		)

		for q in multiturn_qs[1:]:
			run_cmd(
				[
					python,
					script,
					"--mode",
					"multiturn",
					"--history-file",
					history_file,
					"--history-turns",
					"2",
					"-q",
					q,
				]
			)
	else:
		print("\n[skip] multiturn.questions is empty in azure_openai_questions.json")

	# 3) stream with inline context
	if stream_q:
		if stream_ctx:
			stream_input = f"SYSTEM: {stream_ctx} || {stream_q}"
		else:
			stream_input = stream_q
		run_cmd([python, script, "--mode", "stream", "-q", stream_input])
	else:
		print("\n[skip] stream.question is empty in azure_openai_questions.json")

	print("\nAll three modes ran successfully.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())