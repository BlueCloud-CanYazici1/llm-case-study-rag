# ------------------------------------------------------------
# PDF Parsing Utilities
#
# This module extracts text from the source diary PDF and
# converts it into a structured page-level representation.
#
# Each page stores both raw extracted text and a cleaned
# version used later in the chunking pipeline.
# ------------------------------------------------------------

import json
from pypdf import PdfReader

from src.utils import clean_text


def extract_pages_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        raw_text = text if text else ""

        pages.append(
            {
                "page_number": page_number,
                "raw_text": raw_text,
                "clean_text": clean_text(raw_text),
            }
        )

    return pages


def save_pages_to_json(pages, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)