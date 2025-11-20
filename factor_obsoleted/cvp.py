import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_cvp


class CCfgFactorCVP(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args[0]


class CAlgFactorCVP(CAlgFactor):
    def __init__(self, cfg: CCfgFactorCVP):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        iv = instru_data_1m.groupby(by="trade_day")["ret"].apply(lambda z: z.std() * 1e4)
        self.parse_index_to_datetime(iv)
        self.add_to_factor(factor_data, "iv", code, iv)

    def cal_factor(self, *args, iv: pd.DataFrame, prc: pd.DataFrame, vol: pd.DataFrame, **kwargs) -> pd.Series:
        win, lbd = self.cfg.args
        return cal_cvp(iv, prc=prc, vol=vol, win=win, lbd=lbd)
