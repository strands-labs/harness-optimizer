"""Fixtures for the synthetic search tests. The doubles themselves live in helpers.py
so that a test module can import them directly.
"""

import pytest

from tests.search.helpers import DictFormula, ScoreReward


@pytest.fixture
def formula():
    return DictFormula({"document": ""})


@pytest.fixture
def reward():
    return ScoreReward()
