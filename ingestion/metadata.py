"""Extracts metadata (folder, tags, wikilinks, creation date) from loaded documents."""

from ingestion.loader import document_loader
from settings import VAULT_PATH, RIFLESSIONI_FOLDER
from pathlib import Path
import re
import frontmatter
import os
from datetime import datetime

TAG_PATTERN = r"#[\w/]+"
HEX_COLOR_PATTERN = r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$"  # e.g. #FFF3A3A6 or #FFFFFF
WIKILINK_PATTERN = r"\[\[([^\]|#]+)"  # captures the note name, ignores alias (|) and anchor (#)

# Daily note filename: DD-MM-YY (e.g. "09-07-26")
DAILY_NOTE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{2}$")


def is_valid_tag(tag):
    return not re.match(HEX_COLOR_PATTERN, tag)


def estrai_wikilinks(testo: str) -> list[str]:
    return [link.strip() for link in re.findall(WIKILINK_PATTERN, testo)]


def creation_date_da_nome(source_path) -> int | None:
    """For a daily note (DD-MM-YY filename) returns the date as YYYYMMDD
    parsed from the filename, since that's more authoritative than the
    filesystem birthtime (a note written the evening before still refers
    to the next day). Returns None if the filename isn't a valid date,
    so the caller falls back to filesystem birthtime."""
    stem = Path(source_path).stem
    if not DAILY_NOTE_PATTERN.match(stem):
        return None
    try:
        d = datetime.strptime(stem, "%d-%m-%y")
    except ValueError:
        return None  # looks like a date but isn't, e.g. "32-01-26"
    return int(d.strftime("%Y%m%d"))


def estrai_metadati_documenti(documents):
    for doc in documents:
        source_path = doc.metadata["source"]

        doc.metadata["cartella"] = Path(source_path).parent.name

        with open(source_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        body_tags = [
            t.lstrip("#") for t in re.findall(TAG_PATTERN, raw_text) if is_valid_tag(t)
        ]

        post = frontmatter.load(source_path)
        fm_tags = post.get("tags", []) or []

        doc.metadata["tags"] = list(dict.fromkeys(body_tags + fm_tags))

        links = estrai_wikilinks(raw_text)
        doc.metadata["outgoing_links"] = ", ".join(links) if links else "nessun_link"

        # Daily notes in RIFLESSIONI_FOLDER get their date from the filename;
        # every other note uses filesystem birthtime.
        cd_da_nome = (
            creation_date_da_nome(source_path)
            if doc.metadata["cartella"] == RIFLESSIONI_FOLDER
            else None
        )

        if cd_da_nome is not None:
            doc.metadata["creation_date"] = cd_da_nome
        else:
            stat = os.stat(source_path)
            doc.metadata["creation_date"] = int(
                datetime.fromtimestamp(stat.st_birthtime).strftime("%Y%m%d")
            )

    return documents


if __name__ == "__main__":
    # run with: python -m ingestion.metadata
    documents = document_loader(VAULT_PATH)
    documents = estrai_metadati_documenti(documents)
    print(documents[7].metadata)
