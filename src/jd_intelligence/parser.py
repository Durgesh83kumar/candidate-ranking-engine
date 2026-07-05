import os
from src.jd_intelligence.exceptions import JDParserError

try:
    import docx
    _has_docx = True
except ImportError:
    _has_docx = False

class JDParser:
    """Parses raw job description files (DOCX, TXT) into sanitized plain text."""

    def parse(self, file_path: str) -> str:
        """Parses the document at file_path and returns the text content.
        
        Args:
            file_path (str): Path to the job description file.
            
        Returns:
            str: Sanitized plain text content.
        """
        if not os.path.exists(file_path):
            raise JDParserError(f"Job description file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".docx":
            if not _has_docx:
                raise JDParserError("Cannot parse .docx file. python-docx is not installed.")
            return self._parse_docx(file_path)
        elif ext in [".txt", ".md"]:
            return self._parse_txt(file_path)
        else:
            # Fallback to reading as text if extension is unknown
            try:
                return self._parse_txt(file_path)
            except Exception as e:
                raise JDParserError(f"Unsupported file format {ext} and text reading failed: {str(e)}") from e

    def _parse_docx(self, file_path: str) -> str:
        """Parses word document paragraph by paragraph."""
        try:
            doc = docx.Document(file_path)
            paragraphs_text = []
            for p in doc.paragraphs:
                cleaned = p.text.strip()
                if cleaned:
                    paragraphs_text.append(cleaned)
            return "\n".join(paragraphs_text)
        except Exception as e:
            raise JDParserError(f"Failed to parse DOCX file: {str(e)}") from e

    def _parse_txt(self, file_path: str) -> str:
        """Reads plain text files using UTF-8 encoding."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                # Try fall back to cp1252/latin-1
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read().strip()
            except Exception as e:
                raise JDParserError(f"Unicode decode failed for text file: {str(e)}") from e
        except Exception as e:
            raise JDParserError(f"Failed to read text file: {str(e)}") from e
