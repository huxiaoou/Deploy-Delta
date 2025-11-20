import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import robust_div, cal_reoc_by_minute, cal_reoc


class CCfgFactorREOC(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorREOC(CAlgFactor):
    def __init__(self, cfg: CCfgFactorREOC):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        instru_data_1m["doi"] = instru_data_1m["oi"].diff().abs()
        instru_data_1m["eff"] = robust_div(instru_data_1m["doi"], instru_data_1m["vol"], nan_val=0)
        reoc = instru_data_1m.groupby(by="trade_day").apply(cal_reoc_by_minute)
        self.parse_index_to_datetime(reoc)
        self.add_to_factor(factor_data, "reoc", code, reoc)

    def cal_factor(self, *args, reoc: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_reoc(reoc, w0=w0, w1=w1)
