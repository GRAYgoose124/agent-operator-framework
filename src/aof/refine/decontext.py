"""Decontextualisation: make a claim stand alone without changing what it says.

Sentences lifted from papers often lean on their context ("In turn, ...", "Taken together, our findings ...").
A small model rewrites them, and the rewrite is kept only if a judge finds it *supported by its own source
quote*; otherwise the verbatim quote stays the claim. The quote is always preserved in the note's evidence, so a
bad rewrite can never lose or distort the source.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from aof.refine.claims import Claim
from aof.refine.text import needs_context
from aof.specialists import GENERATE, JUDGE, SpecialistExhausted, SpecialistRegistry

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rewrite one sentence from a scientific abstract so that it can be understood on its own. "
    "Replace pronouns, discourse markers (we, our, these, this, in turn, taken together, however...) and missing "
    "subjects with the specific things they refer to, using only the paper title and the sentence. "
    "Do not add facts, numbers or qualifications that are not in the sentence or title. "
    "Write in the third person. Output exactly one sentence and nothing else."
)


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    return text.splitlines()[0].strip() if text else ""


async def rewrite_verified(claim: Claim, registry: SpecialistRegistry) -> str:
    """A standalone rewrite of `claim.text` that a judge found supported by the original, or "" if none."""
    try:
        rewritten = _clean((await registry.call(
            GENERATE, _SYSTEM, f"Paper title: {claim.title}\nSentence: {claim.text}", max_tokens=120,
        )).value)
    except SpecialistExhausted:
        return ""
    if not rewritten or rewritten.lower() == claim.text.lower() or len(rewritten) > 2 * len(claim.text) + 80:
        return ""
    try:
        verdict = (await registry.call(JUDGE, rewritten, f"{claim.title}\n{claim.text}")).value["verdict"]
    except SpecialistExhausted:
        return ""
    if verdict != "supported":
        logger.info("decontext rejected (%s): %r", verdict, rewritten[:80])
        return ""
    return rewritten


async def decontextualize(claim: Claim, registry: SpecialistRegistry) -> Claim:
    """Return `claim` with `statement` set when it opens with a context-dependent marker and a rewrite verifies."""
    if not needs_context(claim.text):
        return claim
    rewritten = await rewrite_verified(claim, registry)
    return replace(claim, statement=rewritten) if rewritten else claim
