"""Hermes Trading Bot - Worker Control.

Manages worker state: running/stopped, version tracking, and graceful shutdown."""
from typing import Dict, Any
import signal
import os
import sys


class WorkerControl:
    """Controls the trading worker lifecycle."""

    def __init__(self):
        self.running = True
        self.strategy_version = "0001"
        self.shutdown_requested = False

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the worker."""
        self.shutdown_requested = True
        self.running = False

    def request_restart(self, new_version: str = None) -> None:
        """Request restart with new strategy version.

        Args:
            new_version: New strategy version to reload
        """
        if new_version:
            self.strategy_version = new_version
        self.running = False

    def signal_handler(self, signum, frame) -> None:
        """Handle OS signals for graceful shutdown."""
        self.request_shutdown()

    def start_signal_handling(self) -> None:
        """Install signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)