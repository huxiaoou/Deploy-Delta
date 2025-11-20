import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_basis


class CCfgFactorBASIS(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorBASIS(CAlgFactor):
    def __init__(self, cfg: CCfgFactorBASIS):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, basis_rate: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_basis(basis_rate, win=self.cfg.args)
