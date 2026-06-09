import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic import simulate_lob_day

@pytest.fixture
def lob_day():
    return simulate_lob_day(seed=0, date="2024-01-02")
