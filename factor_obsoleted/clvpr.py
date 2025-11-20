import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_clvpr_by_minute, cal_clvpr


class CCfgFactorCLVPR(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorCLVPR(CAlgFactor):
    def __init__(self, cfg: CCfgFactorCLVPR):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        clvpr = instru_data_1m.groupby(by="trade_day").apply(cal_clvpr_by_minute)
        self.parse_index_to_datetime(clvpr)
        self.add_to_factor(factor_data, "clvpr", code, clvpr)

    def cal_factor(self, *args, clvpr: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_clvpr(clvpr=clvpr, win=self.cfg.args)
