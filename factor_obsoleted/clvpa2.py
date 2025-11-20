import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_clvpa2_by_minute, cal_clvpa2


class CCfgFactorCLVPA2(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorCLVPA2(CAlgFactor):
    def __init__(self, cfg: CCfgFactorCLVPA2):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        clvpa2 = instru_data_1m.groupby(by="trade_day").apply(cal_clvpa2_by_minute)
        self.parse_index_to_datetime(clvpa2)
        self.add_to_factor(factor_data, "clvpa2", code, clvpa2)

    def cal_factor(self, *args, clvpa2: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_clvpa2(clvpa2=clvpa2, win=self.cfg.args)
