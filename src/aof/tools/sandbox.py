"""Subprocess sandbox for executing agent-created scripts safely."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Imports that agent-created scripts are NOT allowed to use
DEFAULT_BLOCKED_IMPORTS = frozenset(
    {"os", "subprocess", "socket", "shutil", "ctypes", "importlib", "sys"}
)


class Sandbox:
    """Execute scripts in isolated subprocesses with timeout and safety checks."""

    def __init__(
        self,
        timeout: int = 30,
        blocked_imports: frozenset[str] = DEFAULT_BLOCKED_IMPORTS,
    ) -> None:
        self.timeout = timeout
        self.blocked_imports = blocked_imports

    async def execute_script(self, script_path: Path, args: dict) -> dict:
        """Run a Python script in a subprocess. Args passed as JSON on stdin."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdin_data = json.dumps(args).encode()
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=self.timeout,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace").strip(),
                "stderr": stderr.decode(errors="replace").strip(),
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            proc.kill()  # type: ignore[union-attr]
            return {
                "success": False,
                "error": f"Script timed out after {self.timeout}s",
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
            }

    async def test_code(
        self,
        code: str,
        test_cases: list[dict],
    ) -> dict:
        """Write code to a temp file, run each test case, return results.

        Each test_case has {"input": ..., "expected": ...}.
        Returns {"passed": bool, "count": int, "failures": [...]}.
        """
        # Safety check first
        issues = self.validate_safety(code)
        if issues:
            return {"passed": False, "count": 0, "failures": issues}

        results = {"passed": True, "count": 0, "failures": []}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)

        try:
            for i, case in enumerate(test_cases):
                result = await self.execute_script(tmp_path, case.get("input", {}))

                if not result["success"]:
                    results["failures"].append({
                        "test_index": i,
                        "error": result.get("stderr") or result.get("error", ""),
                    })
                    results["passed"] = False
                    continue

                # Check expected output if provided
                expected = case.get("expected")
                if expected is not None:
                    try:
                        actual = json.loads(result["stdout"])
                    except json.JSONDecodeError:
                        actual = result["stdout"]

                    if actual != expected:
                        results["failures"].append({
                            "test_index": i,
                            "expected": expected,
                            "actual": actual,
                        })
                        results["passed"] = False
                        continue

                results["count"] += 1
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        return results

    async def promote_script(
        self,
        code: str,
        meta: dict,
        scripts_dir: Path,
    ) -> Path | None:
        """Save a tested script to the tools directory with metadata header.

        Args:
            code: The Python source code
            meta: {"name": ..., "description": ..., "parameters": {...}}
            scripts_dir: Where to save (data/tools/)

        Returns the file path or None if validation fails.
        """
        issues = self.validate_safety(code)
        if issues:
            logger.warning("Script promotion rejected: %s", issues)
            return None

        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Build the script with metadata docstring
        params_json = json.dumps(meta.get("parameters", {}))
        header = (
            f'"""\n'
            f'@tool\n'
            f'name: {meta["name"]}\n'
            f'description: {meta["description"]}\n'
            f'parameters: {params_json}\n'
            f'"""\n\n'
        )

        file_path = scripts_dir / f"{meta['name']}.py"
        file_path.write_text(header + code, encoding="utf-8")
        logger.info("Promoted script to %s", file_path)
        return file_path

    def validate_safety(self, code: str) -> list[str]:
        """Check code for dangerous operations. Returns list of issues (empty = safe)."""
        issues: list[str] = []

        for imp in self.blocked_imports:
            # Match: import os, from os import ..., __import__("os")
            patterns = [
                rf"\bimport\s+{re.escape(imp)}\b",
                rf"\bfrom\s+{re.escape(imp)}\b",
                rf'__import__\s*\(\s*["\']({re.escape(imp)})["\']',
            ]
            for pattern in patterns:
                if re.search(pattern, code):
                    issues.append(f"Blocked import: {imp}")
                    break

        # Check for dangerous builtins
        dangerous = ["eval", "exec", "compile", "__import__", "globals", "locals"]
        for fn in dangerous:
            if re.search(rf"\b{fn}\s*\(", code):
                issues.append(f"Dangerous builtin: {fn}")

        return issues
