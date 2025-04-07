import math
import numpy as np
from src.rvs.utils import MvnRV

rv = MvnRV(mus=[2, 3], cov=np.asarray([[1, 0], [0, 16]]))
x = np.asarray([1, 5])

def test_logprob():
    logprob = rv.logprob(x, standard_domain=False)
    assert math.isclose(logprob, -3.849, abs_tol=1e-3)

def test_st_logprob():
    st_logprob = rv.logprob(x, standard_domain=True)
    assert math.isclose(st_logprob, -2.462, abs_tol=1e-3)

