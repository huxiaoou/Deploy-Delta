import pandas as pd
from typedef_factor import CCfgFactor, CAlgFactor
from typedef_factor import TInterFactorData
from solutions.math_tools import cal_npc_by_minute, cal_npls


class CCfgFactorNPLS(CCfgFactor):
    @property
    def lag(self) -> int:
        return max(self.args)


class CAlgFactorNPLS(CAlgFactor):
    def __init__(self, cfg: CCfgFactorNPLS):
        super().__init__(cfg=cfg)

    def update_factor_data_in_pre_trans_form(
        self,
        code: str,
        factor_data: TInterFactorData,
        instru_data_1d: pd.DataFrame,
        instru_data_1m: pd.DataFrame,
    ):
        npc = instru_data_1m.groupby(by="trade_day").apply(cal_npc_by_minute)
        self.parse_index_to_datetime(npc)
        self.add_to_factor(factor_data, "npc", code, npc)

    def cal_factor(self, *args, npc: pd.DataFrame, oi: pd.DataFrame, **kwargs) -> pd.Series:
        w0, w1 = self.cfg.args
        return cal_npls(npc, oi=oi, w0=w0, w1=w1)
