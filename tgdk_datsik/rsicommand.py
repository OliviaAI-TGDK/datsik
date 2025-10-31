# rsicommand.py
import logging
import time
import hashlib

class RSICommand:
    """
    RSI Command Module (TGDK/Olivia)
    - Provides symbolic commands during training loops.
    - Commands can map to PossessorScript, CodeWright, or Trainer events.
    """

    def __init__(self, author="Sean 'M' Tichenor"):
        self.author = author
        self.commands = {}
        self.history = []
        self.seed = self._seed_author(author)
        logging.info(f"[RSICommand] Initialized by {author} (seed={self.seed[:12]}...)")

    def _seed_author(self, author: str):
        raw = (author + str(time.time())).encode()
        return hashlib.sha256(raw).hexdigest()

    def register(self, name: str, fn):
        """Register a new RSI command (maps string → callable)."""
        self.commands[name] = fn
        logging.info(f"[RSICommand] Registered command → {name}")

    def execute(self, name: str, *args, **kwargs):
        """Execute a registered command, storing it in history."""
        if name not in self.commands:
            logging.warning(f"[RSICommand] Unknown command: {name}")
            return None

        logging.info(f"[RSICommand] Executing command: {name}")
        result = self.commands[name](*args, **kwargs)
        self.history.append((time.time(), name, result))
        return result

    def list_commands(self):
        """List all available RSI commands."""
        return list(self.commands.keys())

    def last(self, n=5):
        """Show last N command calls."""
        return self.history[-n:]
