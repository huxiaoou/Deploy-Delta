import os
import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qwidgets import check_and_mkdir
from qtools_sxzq.qdata import CDataDescriptor
from solutions.math_tools import weighted_mean
from solutions.eval import plot_nav


class CSectionReturns(SignalStrategy):
    def __init__(
        self,
        sectors: list[str],
        universe_sector: dict[str, str],
        data_desc_pv: CDataDescriptor,
        data_desc_avlb: CDataDescriptor,
    ):
        self.sectors: list[str]
        self.universe_sector: dict[str, str]
        self.data_desc_pv: CDataDescriptor
        self.data_desc_avlb: CDataDescriptor
        super().__init__(sectors, universe_sector, data_desc_pv, data_desc_avlb)

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.create_factor_table(["opn", "cls", "amt"])

    def on_clock(self):
        avlb = self.avlb.get_dict("avlb")
        amt = self.pv.get_dict("amt_major")
        opn = self.pv.get_dict("pre_opn_ret_major")
        cls = self.pv.get_dict("pre_cls_ret_major")
        mkt_data = pd.DataFrame(
            {
                "avlb": avlb,
                "amt": amt,
                "opn": opn,
                "cls": cls,
            }
        ).fillna(0)
        selected_data = mkt_data.query("avlb > 0")
        selected_data["sector"] = selected_data.index.map(lambda z: self.universe_sector[z])
        res: pd.DataFrame = selected_data.groupby(by="sector").apply(  # type:ignore
            lambda z: weighted_mean(x=z[["opn", "cls"]], wgt=z["amt"])
        )
        res["amt"] = selected_data.groupby(by="sector")["amt"].sum()
        res = res.loc[self.sectors]  # type:ignore
        self.update_factor("opn", res["opn"])
        self.update_factor("cls", res["cls"])
        self.update_factor("amt", res["amt"])

    def get_rets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret_opn = self.buffers["opn"].to_dataframe()
        ret_cls = self.buffers["cls"].to_dataframe()
        return ret_opn, ret_cls


def plot(ret_opn: pd.DataFrame, ret_cls: pd.DataFrame, project_data_dir: str):
    check_and_mkdir(save_dir := os.path.join(project_data_dir, "plots"))
    for ret, ret_data in zip(("opn", "cls"), (ret_opn, ret_cls)):
        ret_data.index = ret_data.index.map(lambda z: z.strftime("%Y%m%d"))
        nav_data = ret_data.cumsum()
        plot_nav(
            nav_data=nav_data,
            xtick_count_min=60,
            ylim=(-0.25, 3.50),
            ytick_spread=0.25,
            fig_name=f"sector_returns.{ret}",
            save_dir=save_dir,
            line_style=["-.", "-"],
        )
    return


def main_process_srets(
    span: tuple[str, str],
    sectors: list[str],
    universe_sector: dict[str, str],
    data_desc_pv: CDataDescriptor,
    data_desc_avlb: CDataDescriptor,
    dst_db: str,
    table_srets: str,
    project_data_dir: str,
):
    cfg = {
        "span": span,
        "codes": sectors,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    srets = CSectionReturns(
        sectors=sectors,
        universe_sector=universe_sector,
        data_desc_pv=data_desc_pv,
        data_desc_avlb=data_desc_avlb,
    )
    srets.set_name("srets")
    mat.add_component(srets)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_srets}"
    create_factor_table(dst_path)
    srets.save_factors(dst_path)

    # --- plot
    ret_opn, ret_cls = srets.get_rets()
    plot(ret_opn=ret_opn, ret_cls=ret_cls, project_data_dir=project_data_dir)
    return 0
