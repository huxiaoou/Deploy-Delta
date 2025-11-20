import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_clvpa_by_minute, cal_clvpa


class CCfgFactorCLVPA(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorCLVPA(CAlgFactor):
    def __init__(self, cfg: CCfgFactorCLVPA):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        clvpa = instru_data_1m.groupby(by="trade_day").apply(cal_clvpa_by_minute)
        self.parse_index_to_datetime(clvpa)
        self.add_to_factor(factor_data, "clvpa", code, clvpa)

    def cal_factor(self, *args, clvpa: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_clvpa(clvpa=clvpa, win=self.cfg.args)
