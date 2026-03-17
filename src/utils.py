# ------------------------------------------------------------
# Text Cleaning Utilities
#
# Provides basic text normalization used during PDF parsing.
# The cleaning step removes formatting artifacts from the PDF
# (extra spaces, tabs, irregular line breaks) so the text can
# be reliably segmented and chunked later in the pipeline.
# ------------------------------------------------------------
import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")

    # Cleans up extra spaces at the end of lines
    text = re.sub(r"[ ]+\n", "\n", text)

    # Reduces excessive spaces between words
    text = re.sub(r"[ ]{2,}", " ", text)

    # Converts single line breaks into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Cleans up any extra spaces that may have been created
    text = re.sub(r"[ ]{2,}", " ", text)

    # Cleans up leading and trailing whitespace
    return text.strip()