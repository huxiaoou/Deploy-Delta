import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_val


class CCfgFactorVAL(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorVAL(CAlgFactor):
    def __init__(self, cfg: CCfgFactorVAL):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, close: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_val(close, w0=w0, w1=w1)
