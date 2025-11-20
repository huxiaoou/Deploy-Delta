import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from solutions.math_tools import cal_liquidity


class CCfgFactorLIQUIDITY(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorLIQUIDITY(CAlgFactor):
    def __init__(self, cfg: CCfgFactorLIQUIDITY):
        super().__init__(cfg=cfg)

    def cal_factor(self, *args, ret: pd.DataFrame, turnover: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_liquidity(ret, turnover=turnover, w=self.cfg.args)
