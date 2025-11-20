import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_roll_return, cal_tres


class CCfgFactorTRES(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args


class CAlgFactorTRES(CAlgFactor):
    def __init__(self, cfg: CCfgFactorTRES):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        ts_raw = instru_data_1d.apply(cal_roll_return, args=("ticker_n", "ticker_d", "cls_n", "cls_d"), axis=1)
        self.add_to_factor(factor_data, "ts_raw", code, ts_raw)

    def cal_factor(self, *args, ret: pd.DataFrame, ts_raw: pd.DataFrame, **kwargs) -> pd.Series:
        return cal_tres(ret=ret, ts_raw=ts_raw, win=self.cfg.args)
