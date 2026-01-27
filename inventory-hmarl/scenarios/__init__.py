"""
Scenarios package.

Exports:
- Scenario: Base scenario class
- get_scenario: Get scenario by name
- SCENARIOS: Registry of available scenarios
"""

from .scenarios import (
    Scenario,
    NormalOperations,
    DemandSpike,
    StrongSeasonality,
    SupplierDelay,
    HighVariability,
    get_scenario,
    SCENARIOS
)

__all__ = [
    'Scenario',
    'NormalOperations',
    'DemandSpike',
    'StrongSeasonality',
    'SupplierDelay',
    'HighVariability',
    'get_scenario',
    'SCENARIOS'
]
