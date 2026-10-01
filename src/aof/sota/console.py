"""Interactive console for the harness: stdin is read on a plain thread (never in the event loop's executor, which
can stall model loading on Windows), and progress lines are printed above the prompt as they happen."""

from __future__ import annotations

import asyncio
import sys
import threading

PROMPT = "sota> "
_EOF = object()


class Console:
    def __init__(self) -> None:
        self._lines: asyncio.Queue = asyncio.Queue()
        self._at_prompt = False

    def start_reader(self) -> None:
        loop = asyncio.get_running_loop()

        def reader() -> None:
            while True:
                try:
                    line = sys.stdin.readline()
                except (OSError, ValueError):
                    line = ""
                if line == "":
                    loop.call_soon_threadsafe(self._lines.put_nowait, _EOF)
                    return
                loop.call_soon_threadsafe(self._lines.put_nowait, line)

        threading.Thread(target=reader, name="aof-sota-stdin", daemon=True).start()

    def print(self, text: str) -> None:
        """Print a line without garbling the prompt (the prompt is redrawn after it)."""
        if self._at_prompt:
            sys.stdout.write("\r" + " " * (len(PROMPT) + 2) + "\r")
        sys.stdout.write(text.rstrip("\n") + "\n")
        if self._at_prompt:
            sys.stdout.write(PROMPT)
        sys.stdout.flush()

    async def readline(self) -> str | None:
        sys.stdout.write(PROMPT)
        sys.stdout.flush()
        self._at_prompt = True
        item = await self._lines.get()
        self._at_prompt = False
        return None if item is _EOF else item.rstrip("\r\n")


async def run_console(harness, console: Console) -> None:
    """Read commands until quit/EOF; each is handled immediately while the scheduler keeps running."""
    console.start_reader()
    console.print("Type 'help' for commands. Plain text steers the current work; a question ending in '?' is answered.")
    while not harness.quit:
        line = await console.readline()
        if line is None:
            harness.quit = True
            break
        reply = await harness.submit(line)
        if reply:
            console.print(reply)
