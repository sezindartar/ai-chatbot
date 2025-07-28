from PyPDF2 import PdfReader
from io import BytesIO
from typing import List


def get_pdf_text(pdf_files: List[BytesIO]) -> str:
    """
        Extract and concatenates text content from a list of PDF files.

        Args:
            pdf_files (List[BytesIO]): List of PDF files in memory (as byte stream).


        Returns:
            str: Combined textual context of all pages from all provided pdf files.

        """
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(BytesIO(pdf_file.read()))
        for page in reader.pages:
            text += page.extract_text()
    return text