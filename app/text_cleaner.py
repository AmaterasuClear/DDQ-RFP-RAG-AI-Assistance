"""Text cleaning and normalization pipeline for ingested documents.

Removes extraction artifacts before chunking and embedding so that
retrieved chunks are clean, well-formed, and professional.
"""

from __future__ import annotations

import re


def clean_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces, strip leading/trailing."""
    # Collapse tabs, non-breaking spaces, multiple spaces/newlines into single space
    text = re.sub(r"[\t\r\n\xa0]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def repair_hyphenated_words(text: str) -> str:
    """Repair words broken across lines with hyphens.

    "re-\nsponse" → "response"
    "defini-\ntion" → "definition"
    """
    # Hyphen + newline/space: merge the broken word
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text


def repair_line_continuations(text: str) -> str:
    """Merge lines that were broken mid-sentence.

    A line that ends with a lowercase letter followed by a newline
    and the next line starts with a lowercase letter is likely a
    mid-sentence line break from PDF/DOCX extraction.
    """
    # Replace newlines between lowercase-ending and lowercase-starting lines
    text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)
    # Also handle list items being broken
    text = re.sub(r"([a-z])\n(\d+\.?\s)", r"\1 \2", text)
    return text


def remove_page_artifacts(text: str) -> str:
    """Remove common page artifacts: numbers, headers, footers."""
    # Standalone page numbers
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    # Common header/footer patterns
    text = re.sub(r"\n(Page\s+\d+(\s+of\s+\d+)?)\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n(Confidential|Internal\s+Use\s+Only|DRAFT)\n", "\n", text, flags=re.IGNORECASE)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation spacing and unicode."""
    # Smart quotes → straight quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # Em/en dashes → simple dash
    text = text.replace("—", " — ").replace("–", " – ")
    # Ellipsis
    text = text.replace("…", "...")
    # Bullet points
    text = text.replace("•", "•")
    return text


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline on extracted document text."""
    text = normalize_punctuation(text)
    text = repair_hyphenated_words(text)
    text = repair_line_continuations(text)
    text = remove_page_artifacts(text)
    text = clean_whitespace(text)
    # Remove leading/trailing non-alphanumeric noise
    text = re.sub(r"^[^a-zA-Z0-9]*", "", text)
    text = re.sub(r"[^a-zA-Z0-9.!?)]*$", "", text)
    return text


def is_corrupted_chunk(text: str) -> bool:
    """Check if a chunk contains obvious extraction corruption.

    Returns True if the chunk should be REJECTED.
    """
    if len(text) < 30:
        return True

    # Starts mid-word (lowercase first letter after no sentence context)
    words = text.split()
    if len(words) >= 2:
        first_word = words[0].strip("""'"([{•-""")
        # If first word starts lowercase and looks like a fragment
        if first_word and first_word[0].islower() and len(first_word) < 6:
            # Check if it looks like a continuation (e.g., "ng", "onse)", "tion")
            if re.match(r"^[a-z]{1,5}[)\]]?$", first_word):
                return True

    # Contains excessive line breaks relative to length
    newline_ratio = text.count("\n") / max(len(text), 1)
    if newline_ratio > 0.05:
        return True

    # Unmatched parentheses or brackets (simple heuristic)
    open_count = text.count("(") + text.count("[")
    close_count = text.count(")") + text.count("]")
    if abs(open_count - close_count) > 2:
        return True

    return False