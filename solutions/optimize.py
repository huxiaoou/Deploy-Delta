import numpy as np
import pandas as pd
from typing import Literal, Union
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from transmatrix.event.scheduler import PeriodScheduler
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgOptimizer


def cal_linear_cov(x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    n = len(x)
    z = np.arange(n)
    zxb = z @ x
    zb = z.mean()
    xb = x.mean()
    return zxb - zb * xb


def to_diag_df(v: np.ndarray, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data=np.diag(v), index=names, columns=names)


class COptimizerSecWgt(SignalStrategy):
    CONST_SAFE_RET_LENGTH = 10
    CONST_ANNUAL_FAC = 250

    def __init__(
        self,
        sectors: list[str],
        factors: list[str],
        tgt_rets: list[str],
        cfg_optimizer: CCfgOptimizer,
        data_desc_srets: CDataDescriptor,
        data_desc_fac_agg: CDataDescriptor,
        data_desc_sim_fac: CDataDescriptor,
    ):
        self.sectors: list[str]
        self.factors: list[str]
        self.tgt_rets: list[str]
        self.cfg_optimizer: CCfgOptimizer
        self.data_desc_srets: CDataDescriptor
        self.data_desc_fac_agg: CDataDescriptor
        self.data_desc_sim_fac: CDataDescriptor
        super().__init__(
            sectors,
            factors,
            tgt_rets,
            cfg_optimizer,
            data_desc_srets,
            data_desc_fac_agg,
            data_desc_sim_fac,
        )
        p = len(self.sectors)
        self.opt_val: dict[str, pd.Series] = {
            tgt_ret: pd.Series(np.zeros(p), index=self.sectors) for tgt_ret in self.tgt_rets
        }
        self.snapshots: dict[str, dict] = {tgt_ret: {} for tgt_ret in self.tgt_rets}

    def sim_codes(self, ret: str) -> list[str]:
        return [f"{factor}-{ret}" for factor in self.factors]

    def get_fac_agg(self, periods: int) -> pd.DataFrame:
        raw_data = pd.concat(self.fac_agg.query(self.time, periods=periods))
        raw_data = raw_data.reset_index(level=1, drop=True).T.loc[self.sectors, self.factors]
        return raw_data

    def get_fac_score(self, ret: str, length: int) -> pd.DataFrame:
        data = self.sim_fac.get_window_df(field="net_ret", length=length, codes=self.sim_codes(ret))
        return data

    def init(self):
        # on every day
        self.add_scheduler(milestones="15:00:00", handler=self.on_day_end)

        # on optimizing date
        scheduler = PeriodScheduler(periods="W", milestone="16:00:00")
        self.add_scheduler(scheduler=scheduler, handler=self.on_optimize_date_end)

        # subscribe data
        self.subscribe_data("srets_data", self.data_desc_srets.to_args())
        self.subscribe_data("fac_agg", self.data_desc_fac_agg.to_args())
        self.subscribe_data("sim_fac", self.data_desc_sim_fac.to_args())

        # create factor tables to record factor
        self.create_factor_table(self.tgt_rets)

    def on_day_end(self):
        fac_agg = self.get_fac_agg(periods=1)
        for tgt_ret in self.tgt_rets:
            fac_score = self.get_fac_score(ret=tgt_ret, length=20)
            fac_lcov = cal_linear_cov(x=fac_score.cumsum())
            fac_sign: pd.Series = np.sign(fac_lcov)  # type:ignore
            fac_adj = fac_agg @ to_diag_df(v=fac_sign.to_numpy(), names=self.factors)
            fac_rnk = fac_adj.rank()
            fac_neu = np.sign(fac_rnk - fac_rnk.median())
            fac_wgt = fac_neu / fac_neu.abs().sum()
            sec_wgt = fac_wgt @ pd.Series(data=1, index=self.factors)
            abs_sum = sec_wgt.abs().sum()
            sec_wgt_adj = sec_wgt / (abs_sum if abs_sum > 0 else 1)
            self.update_factor(tgt_ret, sec_wgt_adj[self.sectors])

    def on_optimize_date_end(self):
        pass
        # for tgt_ret in self.tgt_rets:
        #     net_ret_data: pd.DataFrame = self.srets_data.get_window_df(
        #         field=tgt_ret,
        #         length=self.cfg_optimizer.window,
        #         codes=self.sectors,
        #     )
        #     opt_val = self.core(ret_data=net_ret_data, method="sg")
        #     default_val = pd.Series({k: 0 for k in self.sectors})
        #     default_val.update(opt_val)
        #     self.opt_val[tgt_ret] = default_val

    def core(self, ret_data: pd.DataFrame, method: Literal["eq", "sd", "sg"]) -> pd.Series:
        if method == "eq":
            n = ret_data.shape[1]
            return pd.Series(data=np.ones(n) / n, index=self.sectors)
        elif method == "sd":
            sd: pd.Series = ret_data.std()
            w: pd.Series = 1 / sd
            wgt = w / w.abs().sum()
            return wgt[self.sectors]
        elif method == "sg":
            n = ret_data.shape[0]
            w: pd.Series = (np.arange(n) / n) @ ret_data  # type:ignore
            w[w.abs() <= self.cfg_optimizer.lbd] = 0.0
            abs_sum = w.abs().sum()
            wgt = w / (abs_sum if abs_sum > 0 else 1.0)
            return wgt[self.sectors]
        else:
            raise ValueError(f"Invalid method = {method}")


def main_process_optimize_sec_wgt(
    span: tuple[str, str],
    sectors: list[str],
    factors: list[str],
    tgt_rets: list[str],
    cfg_optimizer: CCfgOptimizer,
    data_desc_srets: CDataDescriptor,
    data_desc_fac_agg: CDataDescriptor,
    data_desc_sim_fac: CDataDescriptor,
    dst_db: str,
    table_optimize: str,
):
    """

    :param span:
    :param codes: here codes means product of (factors, returns), like "mtm-cls", "skew-opn", etc.
    :param cfg_factors:
    :param tgt_rets: ["opn", "cls"]
    :param cfg_optimizer:
    :param data_desc_sim:
    :param dst_db: database to save optimized weights for factors
    :param table_optimize: table to save optimized weights for factors
    :return:
    """
    cfg = {
        "span": span,
        "codes": sectors,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    optimizer = COptimizerSecWgt(
        sectors=sectors,
        factors=factors,
        tgt_rets=tgt_rets,
        cfg_optimizer=cfg_optimizer,
        data_desc_srets=data_desc_srets,
        data_desc_fac_agg=data_desc_fac_agg,
        data_desc_sim_fac=data_desc_sim_fac,
    )
    optimizer.set_name("optimizer")
    mat.add_component(optimizer)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_optimize}"
    create_factor_table(dst_path)
    optimizer.save_factors(dst_path)
    return 0
