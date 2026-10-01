"""Concept extraction (tier 0): the recurring, specific terms a vault is about ("place cells", "sleep spindles", "TRN").

Concepts are the navigation backbone of the graph: a claim links to the few concepts it mentions, and a concept note
gathers every claim about it. Extraction is deterministic (no model): candidate phrases are word n-grams that do not
start or end on a stop/generic word, scored by how many claims mention them. A phrase is kept when it is recurring
(`min_df`) but not ubiquitous (`max_share`), and a shorter phrase is dropped when a longer one explains most of its
mentions ("reticular nucleus" inside "thalamic reticular nucleus"). Acronyms defined in the text ("thalamic reticular
nucleus (TRN)") are merged into their phrase, so "TRN" and its expansion are one concept.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aof.refine.text import STOPWORDS

# Words that are frequent in scientific prose but name nothing (never a concept on their own, never a phrase edge).
GENERIC = frozenset("""
study studies result results finding findings evidence role roles effect effects data analysis analyses
increase increases increased decrease decreases decreased change changes changed level levels
process processes function functions mechanism mechanisms type types form forms number numbers
group groups subject subjects participant participants patient patients individual individuals case cases
use used using uses show shows showed shown found find suggest suggests suggested suggesting observed observe
associated association related relationship relationships different difference differences specific important
significant significantly novel new recent previous previously including include includes included however
well known one two three four five first second third several various many further addition
higher lower high low large small greater major minor potential possible likely present absence presence
based involved involve involves involving due whereas while although thus hence therefore
activity activities response responses role process system systems region regions area areas
approach approaches method methods technique techniques experiment experiments model models
task tasks condition conditions factor factors aspect aspects property properties feature features
time times period periods state states way ways term terms part parts set sets range
review paper article work research literature field fields question questions issue issues
cell cells neuron neurons brain structure structures pattern patterns signal signals input inputs output outputs
information level human humans animal animals rat rats mouse mice monkey monkeys
evidence support supports supported demonstrate demonstrated demonstrates reveal revealed reveals
provide provides provided indicate indicates indicated lead leads led contribute contributes contributed
play plays played regulate regulates regulated mediate mediates mediated
across remain remains considered especially general current similar multiple complex critical target influence
understanding thought position domain domains produce promote called reported formed highlight highlights normal
necessary expression population event events dynamic dynamics control formation interaction interactions connection
connections projection projections activation behavior behaviour distinct functional unit units layer layers
underlying increasing decreasing exhibited exhibit exhibits selectively largely whether given within without
recording recordings measure measures measured compared comparison following followed reduced enhanced impaired
impairment impairments deficit deficits disruption loss gain component components mode modes view views account
via occur occurs occurred occurring direct directly ability abilities anterior posterior adult cellular certain
""".split())

# Generic nouns that still make a good *head* of a phrase ("place cells", "memory systems", "grid cell network").
HEAD_OK = frozenset("cell neuron system structure pattern model region area network signal state input output".split())
# Hyphenated modifiers that are fine inside a phrase ("long-term potentiation") but name nothing on their own.
_MODIFIER_TOKENS = frozenset(
    "long-term short-term long-lasting well-known so-called large-scale small-scale real-time high-frequency "
    "low-frequency state-dependent time-dependent activity-dependent age-related".split()
)

# Word endings that mark adjectives/adverbs/participles: such words are never a concept on their own.
_NON_NOUN_ENDINGS = ("al", "ic", "ive", "ous", "ly", "ed", "ble", "ful", "less", "ant", "ary", "tory", "sory", "ing")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
_CHUNK_SPLIT_RE = re.compile(r"[,;:()\[\]{}\"“”!?]|\.\s|\s[-–—]\s")
_ACRONYM_DEF_RE = re.compile(r"((?:[A-Za-z][\w\-]*\s+){1,6}?[A-Za-z][\w\-]*)\s*\(([A-Z][A-Za-z0-9\-]{1,9})\)")


def _is_acronym(tok: str) -> bool:
    return len(tok) >= 2 and (tok.isupper() or (any(c.isdigit() for c in tok) and tok[0].isupper())) and tok.isalnum()


def _norm_word(tok: str) -> str:
    """Case-folded word with a light plural fold (cells -> cell, ripples -> ripple; gyrus/class/analysis stay)."""
    if _is_acronym(tok):
        return tok.upper()
    w = tok.lower().strip("'-").replace("-", " ")  # "sharp-wave ripple" == "sharp wave ripple"
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ys")):
        return w[:-1]
    return w


def _edge_ok(tok: str) -> bool:
    low = tok.lower()
    return (
        low not in STOPWORDS and low not in GENERIC and _norm_word(tok) not in GENERIC
        and len(low) > 1 and not low.isdigit()
    )


def _noun_like(tok: str) -> bool:
    """Crude noun test for single-word concepts (no POS tagger): acronyms, or words without adjective endings."""
    if _is_acronym(tok):
        return True
    low = tok.lower()
    if low.endswith("ent") and not low.endswith("ment"):
        return False
    return len(low) >= 5 and not low.endswith(_NON_NOUN_ENDINGS)


def _chunks(text: str) -> list[list[str]]:
    return [_TOKEN_RE.findall(c) for c in _CHUNK_SPLIT_RE.split(text) if c.strip()]


def candidate_phrases(text: str, max_n: int = 3) -> dict[str, str]:
    """{normalised key: surface form} for every candidate phrase in `text` (each key once)."""
    out: dict[str, str] = {}
    for toks in _chunks(text):
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if not (_edge_ok(gram[0]) and (_edge_ok(gram[-1]) or (n > 1 and _norm_word(gram[-1]) in HEAD_OK))):
                    continue
                inner = [t.lower() for t in gram[1:-1]]
                if any(t in STOPWORDS and t != "of" for t in inner):
                    continue
                if n == 1:
                    tok = gram[0]
                    if not _noun_like(tok) or tok.lower() in _MODIFIER_TOKENS:
                        continue
                if n > 1 and all(t.lower() in GENERIC for t in gram):
                    continue
                key = " ".join(_norm_word(t) for t in gram)
                out.setdefault(key, " ".join(gram))
    return out


def acronym_definitions(texts: Iterable[str], min_votes: int = 2) -> dict[str, str]:
    """{ACRONYM: phrase key} from definitions like "thalamic reticular nucleus (TRN)" whose initials match."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        for m in _ACRONYM_DEF_RE.finditer(text):
            words, acro = m.group(1).split(), m.group(2)
            letters = [c for c in acro.upper() if c.isalpha()]
            # take the shortest tail of the preceding words whose initials spell the acronym's letters
            for k in range(1, len(words) + 1):
                tail = words[-k:]
                initials = "".join(w[0].upper() for w in tail if w.lower() not in STOPWORDS)
                if initials == "".join(letters):
                    key = " ".join(_norm_word(w) for w in tail)
                    votes[acro.upper()][key] += 1
                    break
    # one definition could be a coincidence of initials; require it to recur
    return {acro: key for acro, c in votes.items() for key, n in [c.most_common(1)[0]] if n >= min_votes}


@dataclass
class Concept:
    key: str  # normalised phrase ("place cell")
    label: str  # most common surface form, display case ("place cells")
    aliases: set[str] = field(default_factory=set)  # other keys merged into it (e.g. an acronym)
    claim_ids: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return "concept-" + re.sub(r"[^a-z0-9]+", "-", self.key.lower()).strip("-")[:48]


def _display(surfaces: Counter[str]) -> str:
    form = surfaces.most_common(1)[0][0]
    words = form.split()
    # sentence-initial capitals are noise ("Place cells"); keep acronyms and genuinely capitalised names
    if words and not _is_acronym(words[0]) and words[0][:1].isupper() and words[0][1:].islower():
        lower_votes = sum(c for s, c in surfaces.items() if s.split()[0][:1].islower())
        if lower_votes:
            words[0] = words[0].lower()
    return " ".join(words)


def _kind_rank(key: str) -> int:
    if len(key.split()) > 1:
        return 0
    return 1 if _is_acronym(key) else 2


def extract_concepts(
    items: Sequence[tuple[str, str]],
    *,
    min_df: int = 4,
    max_share: float = 0.06,
    max_concepts: int = 400,
    per_claim: int = 3,
    subsume: float = 0.7,
) -> tuple[list[Concept], dict[str, list[str]]]:
    """Concepts over (claim id, text) items, and each claim's assigned concept keys (most specific first).

    `max_share` caps how common a concept may be: a term in more than that share of claims (the vault's own
    subject) connects everything, which is no signal for navigation.
    """
    n = len(items)
    if n == 0:
        return [], {}
    acronyms = acronym_definitions(t for _, t in items)
    per_item: dict[str, set[str]] = {}
    df: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    for cid, text in items:
        keys = set()
        for key, surface in candidate_phrases(text).items():
            key = acronyms.get(key, key)  # an acronym counts as a mention of its expansion
            keys.add(key)
            surfaces[key][surface] += 1
        per_item[cid] = keys
        df.update(keys)

    limit = max(min_df, int(n * max_share))
    kept = {k: c for k, c in df.items() if min_df <= c <= limit}
    # A shorter phrase mostly explained by a longer one that contains it is redundant ("reticular nucleus").
    by_len = sorted(kept, key=lambda k: -len(k.split()))
    dropped: set[str] = set()
    for long in by_len:
        words = long.split()
        if len(words) < 2:
            continue
        for size in range(1, len(words)):
            for i in range(len(words) - size + 1):
                short = " ".join(words[i:i + size])
                if short in kept and short not in dropped and kept[long] >= subsume * kept[short]:
                    dropped.add(short)
    ranked = sorted(
        (k for k in kept if k not in dropped),
        key=lambda k: (-(kept[k] * (1.0, 0.8, 0.5)[_kind_rank(k)] * (1 + 0.3 * (len(k.split()) - 1))), k),
    )[:max_concepts]
    chosen = set(ranked)

    assigned: dict[str, list[str]] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for cid, _ in items:
        # most specific first: phrases before acronyms before single words, then rarer
        keys = sorted(per_item[cid] & chosen, key=lambda k: (_kind_rank(k), kept[k], k))[:per_claim]
        assigned[cid] = keys
        for k in keys:
            members[k].append(cid)

    concepts = []
    reverse_alias: dict[str, set[str]] = defaultdict(set)
    for acro, key in acronyms.items():
        reverse_alias[key].add(acro)
    for k in ranked:
        if len(members[k]) < max(2, min_df // 2):
            continue
        concepts.append(Concept(k, _display(surfaces[k]), reverse_alias.get(k, set()), members[k]))
    live = {c.key for c in concepts}
    assigned = {cid: [k for k in keys if k in live] for cid, keys in assigned.items()}
    return concepts, assigned


def related_concepts(
    concepts: Sequence[Concept], assigned: dict[str, list[str]], *, top: int = 6, min_pair: int = 2,
) -> dict[str, list[str]]:
    """{concept key: related keys} by co-mention strength (normalised PMI over claims), strongest first."""
    import math

    n = max(1, len(assigned))
    df = {c.key: len(c.claim_ids) for c in concepts}
    pairs: Counter[tuple[str, str]] = Counter()
    for keys in assigned.values():
        ks = sorted(set(keys))
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs[(ks[i], ks[j])] += 1
    scored: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (a, b), c in pairs.items():
        if c < min_pair or a not in df or b not in df:
            continue
        p_ab, p_a, p_b = c / n, df[a] / n, df[b] / n
        npmi = math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)
        scored[a].append((npmi, b))
        scored[b].append((npmi, a))
    return {k: [o for _, o in sorted(v, reverse=True)[:top]] for k, v in scored.items()}
