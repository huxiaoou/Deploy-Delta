import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_minor


class CCfgFactorMINOR(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorMINOR(CAlgFactor):
    def __init__(self, cfg: CCfgFactorMINOR):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret_minor: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_minor(ret_minor, w0=w0, w1=w1)
