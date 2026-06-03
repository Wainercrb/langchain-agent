"""Parser protocols — abstractions for file parsing and parser registry.

Domain code depends on these Protocols instead of concrete infrastructure
parser implementations or the ParserFactory directly.
"""

from pathlib import Path
from typing import Protocol


class FileParser(Protocol):
    """Protocol for file parsers.

    Any object with supports() and parse() methods satisfies this protocol.
    """

    def supports(self, extension: str) -> bool:
        """Check if parser supports file extension."""
        ...

    def parse(self, file_path: Path) -> str:
        """Parse file and return text content."""
        ...


class ParserRegistry(Protocol):
    """Protocol for parser registry/lookup.

    Any object with a get_parser() method satisfies this protocol.
    """

    def get_parser(self, file_extension: str) -> FileParser:
        """Get parser for file extension.

        Args:
            file_extension: File extension (e.g., '.pdf').

        Returns:
            FileParser instance for the given extension.

        Raises:
            Exception if no parser is found for the extension.
        """
        ...
