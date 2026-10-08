"""Loading, cleaning and chunking documentation files.

Owner: Member C

The LangChain docs are MDX (Markdown + JSX). Before chunking we:
  * read the frontmatter title/description
  * drop `import ... from '...'` lines and JSX wrapper tags like <Tip>
  * keep `:::python` blocks and drop `:::js` blocks (Golu is a Python tool)
  * turn links into plain text so they don't pollute embeddings

Chunking is heading-aware: text is first split into sections by Markdown
headings, then packed into chunks of about CHUNK_SIZE characters, never
splitting a code block in half when it can be avoided. Each chunk records
its section path (e.g. "Tools > Create tools > Basic tool definition").

Contextual chunk headers: the text we embed starts with
"Document: <title>\nSection: <section path>". This small trick (also in the
RAG_Techniques repo) helps retrieval when a chunk alone lacks context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOC_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}

_IMPORT_LINE = re.compile(r"^\s*import\s+.+\s+from\s+['\"].+['\"];?\s*$")
_JSX_TAG_LINE = re.compile(r"^\s*</?[A-Z][A-Za-z0-9]*(\s[^>]*)?/?>\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_AT_REF = re.compile(r"@\[([^\]]*)\](\[[^\]]*\])?")
# Self-closing tags that may span lines, e.g. <img src=... /> or <Card ... />
_SELF_CLOSING = re.compile(r"<[A-Za-z][A-Za-z0-9]*\b[^<>]*?/>", re.S)
# Opening/closing JSX component tags (capitalized), anywhere in a line: <Note>, </Tip>
_JSX_TAG_INLINE = re.compile(r"</?[A-Z][A-Za-z0-9.]*(\s[^<>]*)?>")
_FENCE_SPLIT = re.compile(r"(^[ \t]*```.*?^[ \t]*```[^\n]*$)", re.S | re.M)
# Plain HTML layout tags used in the docs: <div style=...>, </div>, <br>, <center>
_HTML_LAYOUT = re.compile(r"</?(div|span|p|br|center|figure|figcaption|section|details|summary)\b[^<>]*>")


@dataclass
class Chunk:
    text: str  # chunk body (what we show to the LLM)
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text used for embeddings and BM25: contextual header + body."""
        header = f"Document: {self.metadata.get('title', '')}\nSection: {self.metadata.get('section', '')}\n\n"
        return header + self.text


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML-ish frontmatter (--- ... ---) from the body. Only simple key: value pairs."""
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip("'\"")
            text = text[end + 4:]
    return meta, text


def clean_mdx(text: str, language: str = "python") -> str:
    """Remove MDX-only syntax and content for other languages."""
    out: list[str] = []
    skipping_lang_block = False
    in_lang_block = False
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if not in_code:
            # :::python / :::js conditional blocks used by the LangChain docs
            if stripped.startswith(":::") and len(stripped) > 3:
                lang = stripped[3:].strip()
                in_lang_block = True
                skipping_lang_block = lang != language
                continue
            if stripped == ":::" and in_lang_block:
                in_lang_block = skipping_lang_block = False
                continue
        if skipping_lang_block:
            continue
        if not in_code and (_IMPORT_LINE.match(line) or _JSX_TAG_LINE.match(line)):
            continue
        out.append(line)

    # Clean prose only; code blocks are kept exactly as written.
    parts = _FENCE_SPLIT.split("\n".join(out))
    for i, part in enumerate(parts):
        if part.lstrip().startswith("```"):
            continue
        part = _SELF_CLOSING.sub("", part)
        part = _JSX_TAG_INLINE.sub("", part)
        part = _HTML_LAYOUT.sub("", part)
        part = _AT_REF.sub(r"\1", part)
        part = _LINK.sub(r"\1", part)
        parts[i] = part
    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def split_sections(text: str, title: str) -> list[tuple[str, str]]:
    """Split Markdown into (section path, body) pairs using headings."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_code = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            path = " > ".join([title] + [h for _, h in stack]) if title else " > ".join(h for _, h in stack)
            sections.append((path, content))
        body.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            body.append(line)
    flush()
    return sections


def _blocks(text: str) -> list[str]:
    """Paragraphs, keeping fenced code blocks whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _hard_split(block: str, size: int) -> list[str]:
    """Split a single oversized block by lines (last resort)."""
    pieces, current = [], ""
    for line in block.splitlines(keepends=True):
        if current and len(current) + len(line) > size:
            pieces.append(current.rstrip())
            current = ""
        current += line
    if current.strip():
        pieces.append(current.rstrip())
    return pieces


def pack_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of ~size chars with ~overlap chars carried over."""
    blocks: list[str] = []
    for block in _blocks(text):
        blocks.extend(_hard_split(block, size) if len(block) > size else [block])

    chunks: list[str] = []
    current: list[str] = []
    for block in blocks:
        if current and len("\n\n".join(current + [block])) > size:
            chunks.append("\n\n".join(current))
            # carry the last paragraphs (up to `overlap` chars) into the next chunk
            carry: list[str] = []
            for prev in reversed(current):
                if len("\n\n".join([prev] + carry)) > overlap:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def doc_url(path: Path, root: Path) -> str:
    """Public URL for a LangChain docs file, or the relative path for other docs."""
    rel = path.relative_to(root).with_suffix("").as_posix()
    if "/src/oss/" in path.as_posix() or root.as_posix().endswith("src/oss/langchain"):
        return f"https://docs.langchain.com/oss/python/langchain/{rel}"
    return rel


def load_and_chunk(docs_dir: Path, size: int, overlap: int, language: str = "python",
                   exclude: tuple[str, ...] = ()) -> list[Chunk]:
    """Load every doc file under docs_dir and return chunks with metadata.

    exclude: relative path prefixes to skip, e.g. ("frontend/", "changelog-js")
    """
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in DOC_EXTENSIONS or not path.is_file():
            continue
        if any(path.relative_to(docs_dir).as_posix().startswith(e) for e in exclude):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(raw)
        title = meta.get("title") or path.stem.replace("-", " ").title()
        body = clean_mdx(body, language)
        for section, section_text in split_sections(body, title):
            for i, piece in enumerate(pack_chunks(section_text, size, overlap)):
                if len(piece.strip()) < 40:  # skip near-empty fragments
                    continue
                chunks.append(Chunk(
                    text=piece,
                    metadata={
                        "source": path.relative_to(docs_dir).as_posix(),
                        "url": doc_url(path, docs_dir),
                        "title": title,
                        "section": section,
                        "part": i,
                    },
                ))
    return chunks
