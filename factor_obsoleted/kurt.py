import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_kurt


class CCfgFactorKURT(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorKURT(CAlgFactor):
    def __init__(self, cfg: CCfgFactorKURT):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_kurt(ret, w0=w0, w1=w1)
