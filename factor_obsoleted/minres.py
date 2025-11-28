import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_minres


class CCfgFactorMINRES(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorMINRES(CAlgFactor):
    def __init__(self, cfg: CCfgFactorMINRES):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, ret_minor: pd.DataFrame, **kwargs) -> pd.Series:
        w = self.cfg.args
        return cal_minres(ret, ret_minor, w=w)
