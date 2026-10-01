"""Console / inbox command language for the harness. Free text is never dropped: a question is answered, anything
else steers the running and following jobs."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

HELP = """\
Ask and write
  ask <question>              answer now from the vault (runs beside the current job)
  report <topic> [-i]         queue a review (-i / --investigation for an insight investigation)
  investigate <question>      queue an insight investigation
    report flags: --sections N  --fresh (rewrite from scratch)  --check none|cheap|strict
  research <question>         gather evidence (papers -> claims) for a question
  deepen <question>           gap research: decompose, find uncovered parts, research them
  now <command>               run a command immediately; the current job pauses and resumes afterwards
Steer
  steer <text>                guidance for the running job (read before its next section) and all later jobs
  <any other text>            same as steer; text ending in '?' is treated as `ask`
  focus <topic> | focus off   bias the autopilot toward a topic
  guidance [clear]            show or clear session guidance
Vault
  map <topic>                 topic hubs and concepts the vault has around a topic
  find <text>                 closest claims
  structure | export | metrics | verify [N] | assess
Queue
  status | jobs | cancel [id] | pause | resume | auto on|off | quit
Models and settings
  models                      roles, context sizes, what is loaded, VRAM plan
  set <key>=<value>           e.g. set writer=large | set writer_ctx=8192 | set check=strict | set sections=9
  config                      all settings ([sota] in config.toml)
"""

COMMANDS = {
    "help", "quit", "exit", "now", "ask", "steer", "focus", "guidance", "map", "find", "status", "jobs", "cancel",
    "pause", "resume", "auto", "set", "config", "models",
    "report", "review", "investigate", "research", "deepen", "structure", "export", "metrics", "verify", "assess",
}
ALIASES = {"?": "help", "q": "quit", "s": "status", "j": "jobs", "r": "report", "a": "ask"}


@dataclass
class Command:
    name: str
    arg: str = ""
    flags: dict = field(default_factory=dict)


def _flags(arg: str) -> tuple[str, dict]:
    """Pull report flags (--sections N, --fresh, --check X, -i/--investigation) out of the argument text."""
    flags: dict = {}
    try:
        tokens = shlex.split(arg, posix=True)
    except ValueError:
        tokens = arg.split()
    rest: list[str] = []
    it = iter(tokens)
    for tok in it:
        if tok in ("-i", "--investigation"):
            flags["kind"] = "investigation"
        elif tok == "--review":
            flags["kind"] = "review"
        elif tok == "--fresh":
            flags["fresh"] = True
        elif tok == "--sections":
            flags["sections"] = int(next(it, "7"))
        elif tok == "--check":
            flags["check"] = next(it, "cheap")
        else:
            rest.append(tok)
    return " ".join(rest), flags


def parse_command(text: str) -> Command:
    text = text.strip()
    head, _, rest = text.partition(" ")
    name = ALIASES.get(head.lower(), head.lower())
    if name in COMMANDS:
        rest = rest.strip()
        if name in ("report", "review", "investigate"):
            rest, flags = _flags(rest)
            return Command(name, rest, flags)
        return Command(name, rest)
    if re.search(r"\?\s*$", text):
        return Command("ask", text)
    return Command("steer", text)
