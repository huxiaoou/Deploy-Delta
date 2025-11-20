import numpy as np
import pandas as pd
from typing import Literal
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table, save_factor
from transmatrix.event.scheduler import PeriodScheduler
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgOptimizer, CCfgFactors


class COptimizerFacWgt(SignalStrategy):
    CONST_SAFE_RET_LENGTH = 10
    CONST_ANNUAL_FAC = 250

    def __init__(
        self,
        cfg_factors: CCfgFactors,
        tgt_rets: list[str],
        cfg_optimizer: CCfgOptimizer,
        data_desc_sim: CDataDescriptor,
    ):
        super().__init__(
            cfg_factors,
            tgt_rets,
            cfg_optimizer,
            data_desc_sim,
        )
        self.factors = cfg_factors.to_list()
        p = len(self.factors)
        self.opt_val: dict[str, pd.Series] = {
            tgt_ret: pd.Series(np.ones(p) / p, index=self.factors) for tgt_ret in self.tgt_rets
        }
        self.snapshots: dict[str, dict] = {tgt_ret: {} for tgt_ret in self.tgt_rets}

    def init(self):
        # on every day
        self.add_scheduler(milestones="15:00:00", handler=self.on_day_end)

        # on optimizing date
        scheduler = PeriodScheduler(periods="W", milestone="16:00:00")
        self.add_scheduler(scheduler=scheduler, handler=self.on_optimize_date_end)

        # subscribe data
        self.subscribe_data("sim_data", self.data_desc_sim.to_args())

        # create factor tables to record factor
        self.create_factor_table(self.tgt_rets)

    def on_day_end(self):
        # print(f"{self.time} for every day")
        for tgt_ret in self.tgt_rets:
            self.snapshots[tgt_ret][self.time] = self.opt_val[tgt_ret]

    def on_optimize_date_end(self):
        # print(f"{self.time} for optimizing date")
        for tgt_ret in self.tgt_rets:
            slc_codes = [f"{fac}-{tgt_ret}" for fac in self.factors]
            net_ret_data: pd.DataFrame = self.sim_data.get_window_df(
                field="net_ret",
                length=self.cfg_optimizer.window,
                codes=slc_codes,
            ).rename(
                columns={k: v for k, v in zip(slc_codes, self.factors)},
            )
            opt_val = self.core(ret_data=net_ret_data, method="sg")
            default_val = pd.Series({k: 0 for k in self.factors})
            default_val.update(opt_val)
            self.opt_val[tgt_ret] = default_val

    def core(self, ret_data: pd.DataFrame, method: Literal["eq", "sd", "sg"]) -> pd.Series:
        if method == "eq":
            n = ret_data.shape[1]
            return pd.Series(data=np.ones(n) / n, index=self.factors)
        elif method == "sd":
            sd: pd.Series = ret_data.std()
            w: pd.Series = 1 / sd
            wgt = w / w.abs().sum()
            return wgt[self.factors]
        elif method == "sg":
            sg: pd.Series = np.sign(ret_data.mean())
            sd: pd.Series = ret_data.std()
            w: pd.Series = sg / sd
            wgt = w / w.abs().sum()
            return wgt[self.factors]
        else:
            raise ValueError(f"Invalid method = {method}")

    def save_to_db(self, table_name: str, db_name: str):
        data = {}
        for tgt_ret in self.tgt_rets:
            data[tgt_ret] = pd.DataFrame.from_dict(
                self.snapshots[tgt_ret],
                orient="index",
            ).stack()
        data = pd.DataFrame(data).reset_index().rename(columns={"level_0": "datetime", "level_1": "code"})
        dst_path = f"{db_name}.{table_name}"
        create_factor_table(dst_path)
        save_factor(table_name=dst_path, data=data)
        return 0


def main_process_optimize_fac_wgt(
    span: tuple[str, str],
    codes: list[str],
    cfg_factors: CCfgFactors,
    tgt_rets: list[str],
    cfg_optimizer: CCfgOptimizer,
    data_desc_sim: CDataDescriptor,
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
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    optimizer = COptimizerFacWgt(
        cfg_factors=cfg_factors,
        tgt_rets=tgt_rets,
        cfg_optimizer=cfg_optimizer,
        data_desc_sim=data_desc_sim,
    )
    optimizer.set_name("optimizer")
    mat.add_component(optimizer)
    mat.init()
    mat.run()

    # --- save
    optimizer.save_to_db(table_optimize, db_name=dst_db)
    return 0
