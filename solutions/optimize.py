import numpy as np
import pandas as pd
from typing import Literal
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from transmatrix.event.scheduler import PeriodScheduler
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgOptimizer


class COptimizerSecWgt(SignalStrategy):
    CONST_SAFE_RET_LENGTH = 10
    CONST_ANNUAL_FAC = 250

    def __init__(
        self,
        sectors: list[str],
        tgt_rets: list[str],
        cfg_optimizer: CCfgOptimizer,
        data_desc_srets: CDataDescriptor,
    ):
        self.cfg_optimizer: CCfgOptimizer
        self.data_desc_srets: CDataDescriptor
        super().__init__(
            sectors,
            tgt_rets,
            cfg_optimizer,
            data_desc_srets,
        )
        self.sectors = sectors
        p = len(self.sectors)
        self.opt_val: dict[str, pd.Series] = {
            tgt_ret: pd.Series(np.zeros(p), index=self.sectors) for tgt_ret in self.tgt_rets
        }
        self.snapshots: dict[str, dict] = {tgt_ret: {} for tgt_ret in self.tgt_rets}

    def init(self):
        # on every day
        self.add_scheduler(milestones="15:00:00", handler=self.on_day_end)

        # on optimizing date
        scheduler = PeriodScheduler(periods="W", milestone="16:00:00")
        self.add_scheduler(scheduler=scheduler, handler=self.on_optimize_date_end)

        # subscribe data
        self.subscribe_data("sret_data", self.data_desc_srets.to_args())

        # create factor tables to record factor
        self.create_factor_table(self.tgt_rets)

    def on_day_end(self):
        for tgt_ret in self.tgt_rets:
            self.update_factor(tgt_ret, self.opt_val[tgt_ret])

    def on_optimize_date_end(self):
        for tgt_ret in self.tgt_rets:
            net_ret_data: pd.DataFrame = self.sret_data.get_window_df(
                field=tgt_ret,
                length=self.cfg_optimizer.window,
                codes=self.sectors,
            )
            opt_val = self.core(ret_data=net_ret_data, method="sg")
            default_val = pd.Series({k: 0 for k in self.sectors})
            default_val.update(opt_val)
            self.opt_val[tgt_ret] = default_val

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
            w: pd.Series = np.arange(n) @ ret_data  # type:ignore
            wgt = w / w.abs().sum()
            return wgt[self.sectors]
        else:
            raise ValueError(f"Invalid method = {method}")


def main_process_optimize_sec_wgt(
    span: tuple[str, str],
    sectors: list[str],
    tgt_rets: list[str],
    cfg_optimizer: CCfgOptimizer,
    data_desc_srets: CDataDescriptor,
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
        tgt_rets=tgt_rets,
        cfg_optimizer=cfg_optimizer,
        data_desc_srets=data_desc_srets,
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
