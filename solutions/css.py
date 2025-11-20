import numpy as np
import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from qtools_sxzq.qdata import CDataDescriptor, save_df_to_db
from typedef import CCfgCss
from solutions.math_tools import weighted_volatility


class CCrossSectionStats(SignalStrategy):
    def __init__(
        self,
        cfg_css: CCfgCss,
        data_desc_pv: CDataDescriptor,
        data_desc_avlb: CDataDescriptor,
    ):
        super().__init__(cfg_css, data_desc_pv, data_desc_avlb)
        self.css: list[dict] = []
        self.last_volatility: list[float] = []

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())

    def on_clock(self):
        self.cfg_css: CCfgCss
        avlb = self.avlb.get_dict("avlb")
        amt = self.pv.get_dict("amt_major")
        ret = self.pv.get_dict("pre_cls_ret_major")
        mkt_data = pd.DataFrame(
            {
                "avlb": avlb,
                "amt": amt,
                "ret": ret,
            }
        ).fillna(0)
        selected_data = mkt_data.query("avlb > 0")
        volatility = weighted_volatility(x=selected_data["ret"], wgt=selected_data["amt"])
        if len(self.last_volatility) >= self.cfg_css.vma_win:
            self.last_volatility.pop(0)
        self.last_volatility.append(volatility)
        vma = np.mean(self.last_volatility)
        tot_wgt = self.cfg_css.vma_wgt if vma >= self.cfg_css.vma_threshold else 1.0
        self.css.append(
            {
                "datetime": self.time,
                "code": "VOL",
                "val": volatility,
            }
        )
        self.css.append(
            {
                "datetime": self.time,
                "code": "VMA",
                "val": vma,
            }
        )
        self.css.append(
            {
                "datetime": self.time,
                "code": "TOTWGT",
                "val": tot_wgt,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.css)


def main_process_css(
    span: tuple[str, str],
    codes: list[str],
    cfg_css: CCfgCss,
    data_desc_pv: CDataDescriptor,
    data_desc_avlb: CDataDescriptor,
    dst_db: str,
    table_css: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    css = CCrossSectionStats(
        cfg_css=cfg_css,
        data_desc_pv=data_desc_pv,
        data_desc_avlb=data_desc_avlb,
    )
    css.set_name("css")
    mat.add_component(css)
    mat.init()
    mat.run()

    # --- save
    save_df_to_db(
        df=css.to_dataframe(),
        db_name=dst_db,
        table_name=table_css,
    )
    return 0
