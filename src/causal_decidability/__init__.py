"""Plan and check discriminating causal experiments.

``decidability``    before the run: can this design separate the declared rivals?
``compatible_set``  after the run: which rivals does the estimate not exclude?
``signatures``      for tabular predictions: which rivals can no outcome of a design separate?
``paired``          after the run: the resolution the per-unit values support for a contrast
"""
from .compatible_set import OUTCOMES, compatible_set, groups_of
from .calculator import (bonferroni_z, decidability, half_ulp, numerical_floor,
                           separation, statistical_floor)
from .design import gains, groups_within, restrict, separating_cells, signatures
from .paired import exact_mcnemar_p, paired, paired_binary, paired_continuous

__all__ = ['OUTCOMES', 'bonferroni_z', 'compatible_set', 'decidability', 'exact_mcnemar_p',
           'gains', 'groups_of', 'groups_within', 'half_ulp', 'numerical_floor', 'paired', 'paired_binary',
           'paired_continuous', 'restrict', 'separating_cells', 'separation', 'signatures',
           'statistical_floor']
__version__ = '0.2.0'
