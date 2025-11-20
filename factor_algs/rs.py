import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_rs


class CCfgFactorRS(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorRS(CAlgFactor):
    def __init__(self, cfg: CCfgFactorRS):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, stock: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_rs(stock=stock, w0=w0, w1=w1)
