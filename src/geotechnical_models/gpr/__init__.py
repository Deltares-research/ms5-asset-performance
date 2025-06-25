"""
Unified GPR module imports.
This module provides a single import point for GPR classes.
"""

try:
    from .gpr_classes import DependentGPRModels, MultitaskGPModel
    __all__ = ["DependentGPRModels", "MultitaskGPModel"]
except ImportError as e:
    print(f"Warning: Could not import GPR classes: {e}")
    # Create dummy classes for compatibility
    class DependentGPRModels:
        def __init__(self):
            raise ImportError("GPR dependencies not available")
    
    class MultitaskGPModel:
        def __init__(self):
            raise ImportError("GPR dependencies not available")
    
    __all__ = []
