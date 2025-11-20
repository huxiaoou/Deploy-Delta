from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgICov


class CFactorICov(SignalStrategy):
    def __init__(self, cfg_icov: CCfgICov, data_desc_pv: CDataDescriptor):
        super().__init__(cfg_icov, data_desc_pv)

    def init(self):
        self.data_desc_pv: CDataDescriptor
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.create_factor_table([_.lower() for _ in self.codes])

    def on_clock(self):
        self.cfg_icov: CCfgICov
        ret = self.pv.get_window_df("pre_cls_ret_major", self.cfg_icov.win)[self.codes]
        icov = ret.fillna(0).cov() * 1e4
        for code in self.codes:
            self.update_factor(code.lower(), icov[code])


def main_process_icov(
    span: tuple[str, str],
    codes: list[str],
    cfg_icov: CCfgICov,
    data_desc_pv: CDataDescriptor,
    dst_db: str,
    table_icov: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    factor_icov = CFactorICov(cfg_icov=cfg_icov, data_desc_pv=data_desc_pv)
    factor_icov.set_name("factor_icov")
    mat.add_component(factor_icov)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_icov}"
    create_factor_table(dst_path)
    factor_icov.save_factors(dst_path)
    return 0
