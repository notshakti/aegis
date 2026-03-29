"""Aegis Environment — simulated workspace and honeytoken infrastructure."""

from .honeytokens import HoneytokenManager
from .workspace import WorkspaceSimulator

__all__ = ["WorkspaceSimulator", "HoneytokenManager"]
