"""ORCHESTRA-WM: synthetic traffic world-model research."""
import os
from pathlib import Path
import tempfile

# Headless cloud containers may have read-only home directories.
_cache = Path(tempfile.gettempdir()) / f'orchestra-wm-{os.getuid() if hasattr(os,"getuid") else "cache"}'
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(_cache / 'cache'))
