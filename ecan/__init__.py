"""
ECAN (Economic Attention Allocation Network) Module

This module implements ECAN-style economic attention allocation and resource 
kernel construction for the OpenCog cognitive architecture.

Key components:
- AttentionKernel: Core 6-dimensional attention tensor management
- EconomicAllocator: Economic attention allocation algorithms  
- ResourceScheduler: Priority queue-based resource scheduling
- AttentionSpreading: Activation spreading with AtomSpace integration
- DecayRefresh: Attention decay and refresh mechanisms
"""

from .attention_kernel import AttentionKernel, ECANAttentionTensor
from .economic_allocator import EconomicAllocator, AttentionAllocationRequest
from .resource_scheduler import ResourceScheduler, ScheduledTask, TaskStatus
from .attention_spreading import AttentionSpreading, AttentionLink
from .decay_refresh import DecayRefresh, DecayParameters, RefreshTrigger, DecayMode

__version__ = "0.1.0"
__author__ = "OpenCog ECAN Team"

__all__ = [
    "AttentionKernel",
    "ECANAttentionTensor", 
    "EconomicAllocator",
    "AttentionAllocationRequest",
    "ResourceScheduler",
    "ScheduledTask",
    "TaskStatus",
    "AttentionSpreading",
    "AttentionLink",
    "DecayRefresh",
    "DecayParameters",
    "RefreshTrigger",
    "DecayMode"
]