from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgAvlb


class CFactorAvlb(SignalStrategy):
    def __init__(self, cfg_avlb: CCfgAvlb, data_desc_pv: CDataDescriptor):
        super().__init__(cfg_avlb, data_desc_pv)

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.create_factor_table(["avlb"])

    def on_clock(self):
        self.cfg_avlb: CCfgAvlb
        amt = self.pv.get_window_df("amt_major", self.cfg_avlb.window)[self.codes]
        amt_aver = amt.mean(axis=0)
        avlb_amt = amt_aver > self.cfg_avlb.threshold

        vol = self.pv.get_window_df("volume_major", self.cfg_avlb.keep)[self.codes]
        vol_sum = (vol.fillna(0) > 0).sum(axis=0)
        avlb_vol = vol_sum >= min(self.cfg_avlb.keep * 0.95, len(vol))

        avlb_tag = avlb_amt & avlb_vol
        avlb = avlb_tag.astype(int)
        self.update_factor("avlb", avlb)


def main_process_avlb(
    span: tuple[str, str],
    codes: list[str],
    cfg_avlb: CCfgAvlb,
    data_desc_pv: CDataDescriptor,
    dst_db: str,
    table_avlb: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    factor_avlb = CFactorAvlb(cfg_avlb=cfg_avlb, data_desc_pv=data_desc_pv)
    factor_avlb.set_name("factor_avlb")
    mat.add_component(factor_avlb)
    mat.init()
    mat.run()

    # # --- check
    # print(f"\n{'AVLB':=^60s}")
    # print(factor_avlb.factor_data["avlb"].to_dataframe())

    # --- save
    dst_path = f"{dst_db}.{table_avlb}"
    create_factor_table(dst_path)
    factor_avlb.save_factors(dst_path)
    return 0
