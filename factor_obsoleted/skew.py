import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_skew


class CCfgFactorSKEW(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorSKEW(CAlgFactor):
    def __init__(self, cfg: CCfgFactorSKEW):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_skew(ret=ret, win=self.cfg.args)
