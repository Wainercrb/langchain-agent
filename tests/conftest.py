"""Pytest configuration for langchain-agent tests.

Configures the Python path so tests can import project modules directly.
"""

import sys
from pathlib import Path

# Add project root to Python path so tests can import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
