from PyPDF2 import PdfReader
import docx
import mammoth
from bs4 import BeautifulSoup
from pathlib import Path


def pdf_to_text(file_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def docx_to_text(file_path: str) -> str:
    """Extract text from DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def doc_to_text(file_path: str) -> str:
    """Extract text from legacy DOC file using Mammoth."""
    with open(file_path, "rb") as f:
        result = mammoth.extract_raw_text(f)
        return result.value

def txt_to_text(file_path: str) -> str:
    """Read text from TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def html_to_text(file_path: str) -> str:
    """Extract text from HTML or HTM file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Clean up empty lines
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


FILE_HANDLERS = {
    ".pdf": pdf_to_text,
    ".docx": docx_to_text,
    ".doc": doc_to_text,
    ".txt": txt_to_text,
    ".html": html_to_text,
    ".htm": html_to_text,
}

def convert_to_txt(file_path: str) -> str:
    """
    Convert PDF, DOCX, DOC, TXT, or HTML files to plain text.
    Returns the text content as a string.
    """
    ext = Path(file_path).suffix.lower()
    handler = FILE_HANDLERS.get(ext)
    if not handler:
        raise ValueError(f"Unsupported file type: {ext}")
    return handler(file_path)