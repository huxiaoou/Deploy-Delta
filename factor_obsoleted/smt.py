import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import preprocess_smt, cal_smt_by_minute, cal_smt


class CCfgFactorSMT(CCfgFactor):
    @property
    def lag(self) -> int:
        return self.args[0]


class CAlgFactorSMT(CAlgFactor):
    def __init__(self, cfg: CCfgFactorSMT):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        _, lbd = self.cfg.args
        slc_data = preprocess_smt(instru_data_1m)
        smt = slc_data.groupby(by="trade_day", group_keys=False).apply(cal_smt_by_minute, lbd=lbd)  # type:ignore
        self.parse_index_to_datetime(smt)
        self.add_to_factor(factor_data, "smt", code, smt)

    def cal_factor(self, *args, smt: pd.DataFrame, **kwargs) -> pd.Series:
        win, _ = self.cfg.args
        return cal_smt(smt, win=win)
