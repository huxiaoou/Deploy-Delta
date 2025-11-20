import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_bres


class CCfgFactorBRES(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorBRES(CAlgFactor):
    def __init__(self, cfg: CCfgFactorBRES):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, basis_rate: pd.DataFrame, **kwargs) -> pd.Series:
        ws, wl = self.cfg.args
        return cal_bres(ret, basis_rate=basis_rate, ws=ws, wl=wl)
