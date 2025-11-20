import os
import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from qtools_sxzq.qwidgets import check_and_mkdir
from qtools_sxzq.qdata import CDataDescriptor, save_df_to_db
from solutions.math_tools import weighted_mean
from solutions.eval import plot_nav


class CSectionReturns(SignalStrategy):
    def __init__(
        self,
        universe_sector: dict[str, str],
        data_desc_pv: CDataDescriptor,
        data_desc_avlb: CDataDescriptor,
    ):
        self.universe_sector: dict[str, str]
        super().__init__(universe_sector, data_desc_pv, data_desc_avlb)
        self.srets: list[pd.DataFrame] = []

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())

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
        res["datetime"] = self.time
        self.srets.append(res)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.concat(self.srets, axis=0, ignore_index=False)
        df = df.reset_index().rename(columns={"sector": "code"})
        df = df[["datetime", "code", "opn", "cls"]]
        return df


def plot(df: pd.DataFrame, project_data_dir: str):
    check_and_mkdir(save_dir := os.path.join(project_data_dir, "plots"))
    for ret in ("opn", "cls"):
        ret_data = pd.pivot_table(data=df, index="datetime", columns="code", values=ret)
        ret_data.index = ret_data.index.map(lambda z: z.strftime("%Y%m%d"))
        nav_data = ret_data.cumsum()
        yl, yu = nav_data.min().min(), nav_data.max().max()
        yl, yu = (int(yl / 0.25) - 1) * 0.25, (int(yu / 0.25) + 1) * 0.25
        plot_nav(
            nav_data=nav_data,
            xtick_count_min=60,
            ylim=(yl, yu),
            ytick_spread=0.25,
            fig_name=f"sector_returns.{ret}",
            save_dir=save_dir,
            line_style=["-.", "-"],
        )
    return


def main_process_srets(
    span: tuple[str, str],
    codes: list[str],
    universe_sector: dict[str, str],
    data_desc_pv: CDataDescriptor,
    data_desc_avlb: CDataDescriptor,
    dst_db: str,
    table_srets: str,
    project_data_dir: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    srets = CSectionReturns(
        universe_sector=universe_sector,
        data_desc_pv=data_desc_pv,
        data_desc_avlb=data_desc_avlb,
    )
    srets.set_name("srets")
    mat.add_component(srets)
    mat.init()
    mat.run()

    # --- save
    df = srets.to_dataframe()
    save_df_to_db(
        df=df,
        db_name=dst_db,
        table_name=table_srets,
    )
    plot(df=df, project_data_dir=project_data_dir)
    return 0
