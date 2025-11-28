import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_mtm


class CCfgFactorMTM(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorMTM(CAlgFactor):
    def __init__(self, cfg: CCfgFactorMTM):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, to_rate: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_mtm(ret, to_rate, w0=w0, w1=w1)
