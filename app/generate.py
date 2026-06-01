from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import GENERATION_PROVIDER
from app.embeddings import tokenize
from app.prompts import JSON_OUTPUT_INSTRUCTIONS, QA_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.schemas import AnswerResponse, Citation, RetrievedChunk

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    references = []
    for chunk in retrieved_chunks:
        references.append({
            "source": chunk.doc_name,
            "page": chunk.page,
            "excerpt": chunk.text,
        })

    return (
        f"QUESTION: {question}\n\n"
        f"REFERENCE DOCUMENTS:\n{json.dumps(references, ensure_ascii=False, indent=2)}\n\n"
        f"{JSON_OUTPUT_INSTRUCTIONS}"
    )


def _call_openrouter_chat(system_prompt: str, user_message: str) -> str:
    """Call OpenRouter chat completions API with retry on rate limits."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    payload = {
        "model": OPENROUTER_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }

    last_error = None
    for attempt in range(6):
        req = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "DDQ-RAG",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32, 64 seconds
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter chat request failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter chat request failed: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            msg = data.get("error", {}).get("message", "no choices returned")
            if "rate" in msg.lower() or "429" in str(data.get("error", {}).get("code", "")):
                last_error = RuntimeError(msg)
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter returned no choices: {msg}")
        return choices[0]["message"]["content"]

    raise RuntimeError(
        f"OpenRouter request failed after 6 retries: {last_error}"
    )


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Call Google Gemini API via REST."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1200,
            "topP": 0.95,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini returned no content parts")
    return parts[0].get("text", "")


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Call Gemini API for answer generation."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return _call_gemini(system_prompt, user_message)


def _clean_json_text(raw: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_response(raw_response: str) -> dict[str, Any]:
    data = json.loads(_clean_json_text(raw_response))
    if not isinstance(data, dict):
        raise ValueError("model response must be a JSON object")
    return data


# ── Local extractive answer synthesis ──────────────────────────────────────────


# ── Section header patterns that indicate raw source artifacts ──────────────────
# These are stripped from chunk text so answers don't parrot document structure.
_SECTION_HEADER_RE = re.compile(
    r"^"
    r"(?:"
    # Multi-word section headers (standalone — consume trailing colon/whitespace)
    r"Primary\s+Regulators|"
    r"Staff\s+Turnover|"
    r"Key\s+Governance|"
    r"General\s+Capabilities|"
    r"Regulatory\s+Environment|"
    r"Corporate\s+Information|"
    r"Operational\s+Statistics|"
    r"Human\s+Resources|"
    r"Legal\s+Entity\s+Name|"
    r"Legal\s+&\s+Head\s+Office\s+Address|"
    r"Head\s+Office\s+Address|"
    r"Company\s+Type|"
    r"Custody\s+Operations\s+Headcount|"
    r"Assets\s+Under\s+Custody|"
    r"Recovery\s+Time\s+Objective|"
    r"Recovery\s+Point\s+Objective|"
    r"Credit\s+Ratings|"
    r"Risk\s+Weighted\s+Assets|"
    r"Basel\s+III\s+Compliance|"
    r"UK\s+FCA\s+Status|"
    r"US\s+SEC\s+Eligibility|"
    r"Network\s+Management|"
    r"Sourcing\s+and\s+Selection|"
    r"On-site\s+Visits|"
    r"Periodic\s+Due\s+Diligence|"
    r"Record\s+Retention|"
    r"Data\s+Breach(?:es)?|"
    r"Sanctions\s+Screening|"
    r"KYC\s+Approach|"
    r"Prohibited\s+Relationships|"
    r"Asset\s+Screening|"
    r"Board\s+Independence|"
    r"Net\s+Zero\s+Target|"
    r"Emissions\s+Data|"
    r"Data\s+Center\s+Architecture|"
    r"Recovery\s+Objectives|"
    r"Pandemic\s+Planning|"
    r"Remote\s+Work|"
    r"Cloud\s+Usage|"
    r"Multi-Factor\s+Authentication|"
    r"System\s+Protection|"
    r"Hardware\s+Security\s+Modules|"
    r"Cyber(?:\s+Security)?\s+(?:Incident|Governance|Framework)|"
    r"Information\s+Security\s+Management\s+System|"
    r"Insolvency\s+Protection|"
    # Acronym-prefixed section labels (consume trailing Eligibility/Compliance word)
    r"(?:US\s+)?SEC\s*(?:Eligibility\s*)?:?\s*|"
    r"ESG\s*(?:&|and)?\s*|"
    r"AML\s*(?:&|and)?\s*|"
    r"CSD\s*(?:Risk\s*)?:?\s*|"
    r"(?:MFA|2FA)\s*:?\s*|"
    r"SIEM\s*(?:&|and)?\s*|"
    r"EDR\s*(?:&|and)?\s*|"
    r"BCP\s*(?:&|and)?\s*|"
    r"DRP?\s*(?:&|and)?\s*|"
    r"GDPR\s*(?:Article\s+\d+)?\s*:?\s*|"
    r"ZTA\s*:?\s*|"
    r"RTO\s*(?:for\s+Cyber\s+Attacks?)?\s*:?\s*|"
    r"RPO\s*:?\s*|"
    r"PEPs?\s*:?\s*|"
    r"EDD\s*:?\s*|"
    r"HSMs?\s*:?\s*|"
    r"MPC\s*(?:/|&|and)?\s*|"
    # Generic prefix words followed by a second word and colon
    r"(?:Regulatory|Corporate|Legal|Risk|Compliance|Governance|"
    r"Security|Insurance|Operations|Financial|Network|Data|IT|"
    r"Cyber|Cloud|Environmental|Social)\s+"
    r"(?:Information|Environment|Management|Framework|Architecture|"
    r"Overview|Policy|Policies|Requirements|Controls|Standards|"
    r"Coverage|Initiatives|Reporting|Data|Selection|Criteria|"
    r"Screening|Retention|Planning|Detection|Response|Protection|"
    r"Usage|Review|Monitoring|Testing|Eligibility|Independence|"
    r"Initiative|Report|Disclosure)"
    r")"
    r"\s*:?\s*",
    re.IGNORECASE,
)

# Patterns that look like data artifacts rather than prose
_FRAGMENT_PATTERNS = [
    re.compile(r"^\d+,\s*(?:Operations|Staff|Employees|Clients|Users):", re.IGNORECASE),
    re.compile(r"^\d+\)\s*$"),  # lone "120)" or "310)"
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}:\s*$"),  # bare "Something Label:"
    re.compile(r"^(?:exceeding|absenteeism|maintains|provides|offers|includes)\s"),
    re.compile(r"^and\s+(?:protocols|procedures|policies|systems)\s+for\b"),
]

# Third-person → first-person substitution pairs
_FIRST_PERSON_REPLACEMENTS = [
    # Organization name references → "we"/"our"
    (re.compile(r"\bClearing\s+Europe\s+(?:S\.?A\.?)\s*(?:maintains?|operates?|offers?|provides?|ensures?)\b", re.IGNORECASE),
     lambda m: "We " + m.group(0).rsplit(None, 1)[-1]),
    (re.compile(r"\bClearing\s+Europe\s+(?:S\.?A\.?)(?:'s|'s)\b", re.IGNORECASE), "our"),
    (re.compile(r"\bClearing\s+Europe\s+(?:S\.?A\.?)\b", re.IGNORECASE), "we"),
    (re.compile(r"\b(?:the|our)\s+(?:Firm|Company|Organization|Group)\s*(?:'s|'s)\b", re.IGNORECASE), "our"),
    (re.compile(r"\b(?:the|our)\s+(?:Firm|Company|Organization|Group)\s+(?:maintains?|operates?|offers?|provides?|ensures?|has|holds?|keeps?|uses?|employs?|mandates?|requires?|conducts?|performs?)\b", re.IGNORECASE),
     lambda m: "We " + m.group(0).rsplit(None, 1)[-1]),
    (re.compile(r"\b(?:the|our)\s+(?:Firm|Company|Organization|Group)\b", re.IGNORECASE), "we"),
    # "Our X policy mandates..." / "Our X framework ensures..."
    (re.compile(r"\b(?:the\s+)?(?:firm'?s?|company'?s?|organization'?s?)\s+(policy|framework|system|process|procedure|approach|team|department)\b", re.IGNORECASE),
     r"our \1"),
    # "According to our X Policy" → keep; "The X Policy states" → "Our X Policy states"
    (re.compile(r"\bThe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s+(Policy|Framework|Plan)\s+(states?|mandates?|requires?|outlines?|defines?|specifies?)\b"),
     r"Our \1 \2 \3"),
    # Staff/employee references
    (re.compile(r"\bthe\s+staff\s+turnover\s+rate\b", re.IGNORECASE), "our staff turnover rate"),
    (re.compile(r"\b(?:the\s+)?(?:firm'?s?|company'?s?)\s+(staff|employees|workforce|team)\b", re.IGNORECASE), r"our \1"),
]


def _strip_section_headers(text: str) -> str:
    """Strip section header prefixes from text. Handles chained headers."""
    text = text.strip()
    # Strip parenthesized acronyms at start: "(RTO):", "(AUC)", etc.
    text = re.sub(r"^\([A-Z]{2,6}\)\s*:?\s*", "", text)
    for _ in range(4):
        new_text = _SECTION_HEADER_RE.sub("", text, count=1).strip()
        if new_text == text:
            break
        text = new_text
        # Also strip parenthesized acronyms that may appear after headers
        text = re.sub(r"^\([A-Z]{2,6}\)\s*:?\s*", "", text)
    return text


def _clean_chunk_text(text: str) -> str:
    """Strip document title prefixes and section headers from chunk text."""
    text = _strip_section_headers(text)

    # Remove leading document title pattern followed by numbered sections
    # e.g. "Asset Safety and Client Money Policy 1. Legal Title ..."
    match = re.match(
        r"^[\w\s&,()/.-]{10,80}?(?:\s+(?=\d+\.\s+[A-Z]))",
        text,
    )
    if match:
        text = text[match.end():].strip()

    # Remove leading bare section numbers like "1. " or "2.1 " or "4. "
    text = re.sub(r"^(?:\d+\.)+\s+", "", text)

    # Remove leading parenthesized acronyms like "(RTO):", "(AUC):", "(RWA)"
    text = re.sub(r"^\([A-Z]{2,6}\)\s*:?\s*", "", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text


def _is_fragment(sentence: str) -> bool:
    """Return True if the sentence looks like a data artifact, not prose."""
    sent = sentence.strip()
    if len(sent) < 15:
        return True
    for pat in _FRAGMENT_PATTERNS:
        if pat.match(sent):
            return True
    # Starts with a bare number followed by comma (data dump artifact)
    if re.match(r"^\d{2,},\s", sent):
        return True
    # Starts with lowercase or parenthesis (orphaned continuation)
    if sent and sent[0].islower():
        return True
    if sent and sent[0] in ")]})":
        return True
    return False


def _to_first_person(text: str) -> str:
    """Rewrite third-person institutional text into first-person 'we'/'our' statements."""
    result = text
    for pattern, replacement in _FIRST_PERSON_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def _sentence_relevance_score(sentence: str, question_tokens: set[str]) -> int:
    """Score how relevant a sentence is to the question.

    Uses expanded token matching: question tokens are stemmed and expanded
    to related forms so that e.g. 'regulatory' matches 'regulators', 'bodies'
    matches 'body', 'oversight' matches 'supervised', etc.
    """
    sent_tokens = set(tokenize(sentence))
    # Direct match
    direct = len(question_tokens & sent_tokens)
    if direct > 0:
        return direct
    # Expanded match — check related word forms
    expanded_hits = 0
    for qt in question_tokens:
        if _token_matches_sentence(qt, sent_tokens):
            expanded_hits += 1
    return expanded_hits


def _stem_word(word: str) -> str:
    """Lightweight stemmer that collapses common word-form variations."""
    w = word.lower()
    # -ies → -y (bodies → body)
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    # -ors → -or (regulators → regulator)
    if len(w) > 5 and w.endswith("ors"):
        return w[:-3] + "or"
    # -ers → -er (providers → provider)
    if len(w) > 5 and w.endswith("ers"):
        return w[:-3] + "er"
    # -ing
    if len(w) > 5 and w.endswith("ing") and not w.endswith("thing"):
        return w[:-3]
    # -ed
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    # -es
    if len(w) > 4 and w.endswith("es") and not w.endswith(("sses", "shes", "ches")):
        return w[:-2]
    # -s (but not -ss, -us, -is)
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


# Common word-form pairs where stemming alone won't bridge the gap
_WORD_FORM_EXPANSIONS: dict[str, set[str]] = {
    "regulatory": {"regulator", "regulators", "regulation", "regulations", "regulated"},
    "regulator": {"regulatory", "regulation", "regulators", "regulated"},
    "regulators": {"regulatory", "regulator", "regulation", "regulated"},
    "oversight": {"supervise", "supervised", "supervises", "supervision", "supervisor",
                  "supervisory", "oversee", "oversees", "overseeing", "monitor",
                  "monitors", "monitoring", "monitored"},
    "supervised": {"oversight", "supervise", "supervision", "supervisor", "supervisory"},
    "bodies": {"body", "authority", "authorities", "agency", "agencies", "institution",
               "institutions", "entity", "entities", "regulator", "regulators"},
    "body": {"bodies", "authority", "authorities", "agency", "agencies", "institution"},
    "responsible": {"responsible", "oversee", "oversees", "supervise", "supervised",
                    "supervision", "administer", "administers", "administered"},
    "specify": {"specifies", "specified", "specification", "specifications", "name",
                "names", "named", "list", "lists", "listed", "identify", "identifies",
                "identified"},
    "attrition": {"turnover", "attrition", "retention", "departed", "departure",
                  "departures", "left", "resigned", "resignations"},
    "prudential": {"prudential", "regulatory", "financial", "banking", "supervision",
                   "supervisory"},
}
# Build reverse expansions: if A expands to B, B should expand to A
for _k, _v in list(_WORD_FORM_EXPANSIONS.items()):
    for _word in _v:
        if _word not in _WORD_FORM_EXPANSIONS:
            _WORD_FORM_EXPANSIONS[_word] = set()
        _WORD_FORM_EXPANSIONS[_word].add(_k)


def _token_matches_sentence(question_token: str, sent_tokens: set[str]) -> bool:
    """Check if a question token matches any token in the sentence."""
    qt_stem = _stem_word(question_token)
    for st in sent_tokens:
        st_stem = _stem_word(st)
        if qt_stem == st_stem:
            return True
    # Check expansion mappings
    expansions = _WORD_FORM_EXPANSIONS.get(question_token, set())
    if expansions & sent_tokens:
        return True
    # Also check expansions against stemmed forms
    for exp in expansions:
        if _stem_word(exp) in {_stem_word(st) for st in sent_tokens}:
            return True
    return False


def _is_proper_sentence(sent: str) -> bool:
    """Return True if the sentence looks like a complete, well-formed English sentence."""
    sent = sent.strip()
    if len(sent) < 15:
        return False
    # Must start with a capital letter or "we"/"our" variant
    if not (sent[0].isupper() or sent[0] in '"\'('):
        return False
    # Must end with sentence-ending punctuation
    if sent[-1] not in ".!?":
        return False
    return True


def _build_extractive_answer(
    chunks: list[RetrievedChunk],
    question_tokens: set[str],
    max_chars: int = 1200,
) -> str:
    """Build a first-person, professional answer from retrieved chunks.

    Steps:
    1. Take top chunks by similarity, clean each one.
    2. Select sentences relevant to the question, discard fragments.
    3. Rewrite selected sentences into first-person ("we"/"our") form.
    4. If no sentence opens with "We", prepend a bridging opener.
    5. Join into flowing paragraphs, cut cleanly at sentence boundaries.
    """
    top_chunks = chunks[:5]

    # ── Collect and score all sentences across top chunks ─────────────────
    scored_sentences: list[tuple[int, str]] = []  # (score, sentence)
    seen: set[str] = set()

    for chunk in top_chunks:
        text = _clean_chunk_text(chunk.text)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            sent = sent.strip()
            # Strip section headers from the start of each sentence too
            sent = _strip_section_headers(sent)
            if _is_fragment(sent):
                continue
            if not _is_proper_sentence(sent):
                continue
            # De-duplicate near-identical sentences
            key = sent[:60].lower()
            if key in seen:
                continue
            seen.add(key)

            score = _sentence_relevance_score(sent, question_tokens)
            if score > 0:
                scored_sentences.append((score, sent))

    # Sort by relevance score descending
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # ── Fallback: if no sentence scored > 0, vocabulary mismatch is likely ──
    # (e.g. question asks about "regulatory bodies" but doc says "Primary Regulators")
    # Include all proper sentences from top chunks as a fallback.
    if not scored_sentences or scored_sentences[0][0] == 0:
        for chunk in top_chunks[:3]:
            text = _clean_chunk_text(chunk.text)
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if _is_fragment(sent):
                    continue
                if not _is_proper_sentence(sent):
                    continue
                key = sent[:60].lower()
                if key in seen:
                    continue
                seen.add(key)
                scored_sentences.append((1, sent))

    # ── Build answer from best sentences ──────────────────────────────────
    selected: list[str] = []
    total_chars = 0

    for _score, sent in scored_sentences:
        # Convert to first-person
        transformed = _to_first_person(sent)
        # Capitalize first letter
        if transformed and transformed[0].islower():
            transformed = transformed[0].upper() + transformed[1:]

        if total_chars + len(transformed) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                cut = transformed[:remaining]
                # Find last complete sentence boundary
                for punct in (". ", "? ", "! "):
                    last = cut.rfind(punct)
                    if last > 60:
                        transformed = cut[:last + 1]
                        break
                else:
                    last_period = cut.rfind(". ")
                    if last_period > 40:
                        transformed = cut[:last_period + 1]
                    else:
                        continue
            else:
                break

        selected.append(transformed)
        total_chars += len(transformed)

    if not selected:
        # Fallback: take first proper sentence from the best chunk, cleaned
        for chunk in top_chunks:
            text = _clean_chunk_text(chunk.text)
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 25 and sent[0].isupper() and sent[-1] in ".!?":
                    selected.append(_to_first_person(sent))
                    break
            if selected:
                break
        if not selected and top_chunks:
            # Last resort
            raw = top_chunks[0].text.strip()[:600]
            selected.append(_to_first_person(raw))

    answer = " ".join(selected).strip()

    # ── Ensure answer opens with "We" ─────────────────────────────────────
    if answer and not re.match(r"^\s*(?:We|Our)\b", answer, re.IGNORECASE):
        if answer[0].isupper():
            answer = "We " + answer[0].lower() + answer[1:]
        else:
            answer = "We " + answer
    else:
        # Capitalize first letter if it's lowercase after "Our"/"We" prefix
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]

    # ── Clean trailing garbage at sentence boundaries ─────────────────────
    # If the last sentence is incomplete (no ending punctuation), trim it
    last_punct = max(answer.rfind("."), answer.rfind("?"), answer.rfind("!"))
    if last_punct > len(answer) * 0.7 and last_punct < len(answer) - 1:
        # There's text after the last punctuation that may be garbage
        after = answer[last_punct + 1:].strip()
        if len(after) < 15 or after[0].islower() or not after[0].isalpha():
            answer = answer[:last_punct + 1]

    if answer and answer[-1] not in ".!?":
        # Cut back to the last complete sentence
        for punct in (".", "?", "!"):
            last = answer.rfind(punct)
            if last > len(answer) * 0.6:
                answer = answer[:last + 1]
                break

    return answer


def _extract_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Extract a coherent answer from retrieved chunks."""
    return _build_extractive_answer(retrieved_chunks, set(tokenize(question)))


def _no_evidence_response(question: str) -> str:
    """Return a professional refusal when no relevant evidence exists."""
    return (
        "We are unable to provide a substantiated response to this inquiry "
        "based on the due diligence materials currently indexed in our knowledge "
        "base. Our uploaded policy documents do not contain specific information "
        "that directly addresses this question. We recommend submitting the "
        "relevant documentation or reaching out to our relationship management "
        "team for further assistance."
    )


def _local_model_response(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    top_similarity: float = 0.0,
) -> AnswerResponse:
    """Build an answer from retrieved evidence without an external LLM.

    Returns a professional extractive answer when evidence is available,
    or a clear refusal when no relevant evidence exists.
    """
    if not retrieved_chunks:
        return AnswerResponse(
            answer=_no_evidence_response(question),
            has_source=False,
            confidence_level="LOW",
            uncertainty_note="No relevant document chunks were found in our knowledge base.",
            source_citations=[],
        )

    # Require at least one chunk above the similarity threshold
    from app.config import SIMILARITY_THRESHOLD as _SIM_THRESHOLD

    best_similarity = retrieved_chunks[0].similarity if retrieved_chunks else 0.0
    if best_similarity < _SIM_THRESHOLD:
        return AnswerResponse(
            answer=_no_evidence_response(question),
            has_source=False,
            confidence_level="LOW",
            uncertainty_note=f"Our highest similarity score ({best_similarity:.2f}) was below the evidence threshold.",
            source_citations=[],
        )

    answer = _extract_answer(question, retrieved_chunks)
    return AnswerResponse(
        answer=answer,
        has_source=True,
        confidence_level="MEDIUM",
        uncertainty_note="",
        source_citations=[],
    )


# ── Main generation entry point ────────────────────────────────────────────────


def inject_citations(
    model_output: AnswerResponse,
    retrieved_chunks: list[RetrievedChunk],
) -> AnswerResponse:
    citations = [
        Citation(
            doc_name=chunk.doc_name,
            page=chunk.page,
            quote=chunk.text[:200],
            similarity=chunk.similarity,
        )
        for chunk in retrieved_chunks
    ]
    return AnswerResponse(
        answer=model_output.answer,
        has_source=model_output.has_source and bool(citations),
        confidence_level=model_output.confidence_level,
        uncertainty_note=model_output.uncertainty_note,
        source_citations=citations,
    )


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    provider: str | None = None,
) -> AnswerResponse:
    active_provider = (provider or GENERATION_PROVIDER).lower()

    if active_provider in ("gemini", "openrouter"):
        prompt = build_prompt(question, retrieved_chunks)
        try:
            raw = _call_llm(SYSTEM_PROMPT, prompt)
            parsed = parse_response(raw)
            response = AnswerResponse(
                answer=parsed.get("answer", ""),
                has_source=parsed.get("has_evidence", True),
                confidence_level=parsed.get("confidence", "MEDIUM").upper(),
                uncertainty_note=parsed.get("limitations", ""),
                source_citations=[],
            )
        except Exception as exc:
            # LLM unavailable — use local extractive synthesis
            top_sim = retrieved_chunks[0].similarity if retrieved_chunks else 0.0
            response = _local_model_response(question, retrieved_chunks, top_sim)
            if response.has_source:
                response = AnswerResponse(
                    answer=response.answer,
                    has_source=True,
                    confidence_level="MEDIUM",
                    uncertainty_note="Answer was derived from our indexed policy documents using extractive synthesis.",
                    source_citations=[],
                )
    else:
        top_sim = retrieved_chunks[0].similarity if retrieved_chunks else 0.0
        response = _local_model_response(question, retrieved_chunks, top_sim)

    return inject_citations(response, retrieved_chunks)


def generate_qa_batch(
    questions: list[str],
    retrieved_for_each: list[list[RetrievedChunk]],
) -> list[dict[str, Any]]:
    """Generate answers for multiple questions in a single API call."""
    qa_items = []
    for idx, (question, chunks) in enumerate(zip(questions, retrieved_for_each)):
        references = []
        for chunk in chunks:
            references.append({
                "source": chunk.doc_name,
                "page": chunk.page,
                "excerpt": chunk.text,
            })
        qa_items.append({"index": idx, "question": question, "references": references})

    user_message = (
        "Answer each due diligence question below using only the provided reference excerpts.\n\n"
        f"QUESTIONNAIRE ITEMS:\n{json.dumps(qa_items, ensure_ascii=False, indent=2)}\n\n"
        'For each question, return JSON: {"index": N, "question": "...", "answer": "...", '
        '"has_evidence": bool, "confidence": "HIGH"/"MEDIUM"/"LOW", "limitations": ""}\n'
        "Return a JSON array. No markdown fences, no preamble."
    )

    try:
        raw = _call_llm(QA_SYSTEM_PROMPT, user_message)
        cleaned = _clean_json_text(raw)
        results = json.loads(cleaned)
        if not isinstance(results, list):
            raise ValueError("Expected a JSON array")
        return results
    except Exception:
        results = []
        for question, chunks in zip(questions, retrieved_for_each):
            try:
                result = generate_answer(question, chunks)
                results.append({
                    "index": len(results),
                    "question": question,
                    "answer": result.answer,
                    "has_evidence": result.has_source,
                    "confidence": result.confidence_level,
                    "limitations": result.uncertainty_note,
                })
            except Exception:
                results.append({
                    "index": len(results),
                    "question": question,
                    "answer": "Unable to generate answer.",
                    "has_evidence": False,
                    "confidence": "LOW",
                    "limitations": "Generation error",
                })
        return results


def save_raw_response_for_review(raw_response: str, destination: str | Path) -> None:
    Path(destination).write_text(raw_response, encoding="utf-8")