"""
Temporary compatibility layer for old import patterns.
This helps during the transition period.
"""

# Make old import patterns work temporarily
import sys
from pathlib import Path

# Add current src to path for compatibility
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
