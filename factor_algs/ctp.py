import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_ctp


class CCfgFactorCTP(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args[0]


class CAlgFactorCTP(CAlgFactor):
    def __init__(self, cfg: CCfgFactorCTP):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, to_rate: pd.DataFrame, prc: pd.DataFrame, vol: pd.DataFrame, **kwargs) -> pd.Series:
        win, lbd = self.cfg.args
        return cal_ctp(to_rate, prc=prc, vol=vol, win=win, lbd=lbd)
