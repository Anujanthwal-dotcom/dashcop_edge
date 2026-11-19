"""Violations module for state tracking, best-shot selection, and e-ticketing."""

from .best_shot import BestShotSelector
from .state_manager import ViolationStateManager
from .ticket_dispatcher import TicketDispatcher

__all__ = ["BestShotSelector", "ViolationStateManager", "TicketDispatcher"]
