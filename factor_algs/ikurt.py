import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_ikurt


class CCfgFactorIKURT(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorIKURT(CAlgFactor):
    def __init__(self, cfg: CCfgFactorIKURT):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        ikurt = instru_data_1m.groupby(by="trade_day")["ret"].apply(lambda z: (z * 1e4).kurt())
        self.parse_index_to_datetime(ikurt)
        self.add_to_factor(factor_data, "ikurt", code, ikurt)

    def cal_factor(self, *args, ikurt: pd.DataFrame, **kwargs) -> pd.Series:
        win = self.cfg.args
        return cal_ikurt(ikurt, win=win)
