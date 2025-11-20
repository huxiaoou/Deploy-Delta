import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef_factor import CCfgFactors
from solutions.math_tools import gen_exp_wgt


def map_factor_to_signal(nrm_data: pd.DataFrame) -> pd.DataFrame:
    """

    :param nrm_data: index = code_name, like "AG9999_SHFE".
                columns = ["avlb", "nrm"]
    :return: nrm_data, a pd.Dataframe, index = code_name,
                columns = ["avlb", "nrm", "weight"]
    """
    slc_data = nrm_data.query("avlb > 0")
    slc_data = slc_data.sort_values(by=["nrm", "instru"], ascending=[False, True])
    slc_data = slc_data.dropna(subset=["nrm"], axis=0)
    n = len(slc_data)
    slc_data["weight"] = gen_exp_wgt(n)
    nrm_data["weight"] = slc_data["weight"]
    nrm_data["weight"] = nrm_data["weight"].fillna(0)
    return nrm_data[["avlb", "nrm", "weight"]]


"""
--------------------------------
-------- signals factor --------
--------------------------------
"""


class CSignalsFac(SignalStrategy):
    def __init__(self, cfg_factors: CCfgFactors, data_desc_avlb: CDataDescriptor, data_desc_fac_nrm: CDataDescriptor):
        super().__init__(cfg_factors, data_desc_avlb, data_desc_fac_nrm)

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.subscribe_data("fac_nrm", self.data_desc_fac_nrm.to_args())
        self.create_factor_table(self.cfg_factors.to_list())

    def on_clock(self):
        avlb = self.avlb.get_dict("avlb")
        for factor in self.cfg_factors.to_list():
            fac = self.fac_nrm.get_dict(factor)
            nrm_data = pd.DataFrame(
                {
                    "avlb": avlb,
                    "nrm": fac,
                }
            )
            nrm_data.index.name = "instru"
            nrm_data = map_factor_to_signal(nrm_data)
            sorted_data = nrm_data.loc[self.codes]
            self.update_factor(factor, sorted_data["weight"].to_numpy())


def main_process_signals_fac(
    span: tuple[str, str],
    codes: list[str],
    cfg_factors: CCfgFactors,
    data_desc_avlb: CDataDescriptor,
    data_desc_fac_nrm: CDataDescriptor,
    dst_db: str,
    table_sig_fac: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    signals_fac_agg = CSignalsFac(
        cfg_factors=cfg_factors,
        data_desc_avlb=data_desc_avlb,
        data_desc_fac_nrm=data_desc_fac_nrm,
    )
    signals_fac_agg.set_name("signals_fac")
    mat.add_component(signals_fac_agg)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_sig_fac}"
    create_factor_table(dst_path)
    signals_fac_agg.save_factors(dst_path)
    return 0


"""
-----------------------------
------- signals stg ---------
-----------------------------
"""


class CSignalsStg(SignalStrategy):
    def __init__(
        self,
        tgt_rets: list[str],
        sectors: list[str],
        universe_sector: dict[str, str],
        data_desc_pv: CDataDescriptor,
        data_desc_optimize: CDataDescriptor,
        data_desc_css: CDataDescriptor,
        data_desc_avlb: CDataDescriptor,
    ):
        self.sectors: list[str]
        self.universe_sector: dict[str, str]
        super().__init__(
            tgt_rets,
            sectors,
            universe_sector,
            data_desc_pv,
            data_desc_optimize,
            data_desc_css,
            data_desc_avlb,
        )

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.subscribe_data("optimize", self.data_desc_optimize.to_args())
        self.subscribe_data("css", self.data_desc_css.to_args())
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.create_factor_table(self.tgt_rets)

    def on_clock(self):
        tot_wgt: float = self.css.get_dict("val")["TOTWGT"]
        for tgt_ret in self.tgt_rets:
            sec_wgt = pd.DataFrame({"sector_wgt": self.optimize.get_dict(tgt_ret)})
            data = pd.DataFrame(
                {
                    "avlb": self.avlb.get_dict("avlb"),
                    "amt": self.pv.get_dict("amt_major"),
                }
            ).fillna(0)
            data["amt_avlb"] = data["avlb"] * data["amt"]
            data["sector"] = data.index.map(lambda z: self.universe_sector[z])
            data["inner_wgt"] = data.groupby(by="sector")["amt_avlb"].apply(lambda z: z / z.sum())
            data = data.merge(right=sec_wgt, left_on="sector", right_index=True, how="left")
            raw_wgt = data["inner_wgt"] * data["sector_wgt"]
            adj_wgt = raw_wgt / raw_wgt.abs().sum()
            opt_wgt = adj_wgt * tot_wgt
            self.update_factor(tgt_ret, opt_wgt[self.codes].to_numpy())


def main_process_signals_stg(
    span: tuple[str, str],
    codes: list[str],
    tgt_rets: list[str],
    sectors: list[str],
    universe_sector: dict[str, str],
    data_desc_pv: CDataDescriptor,
    data_desc_optimize: CDataDescriptor,
    data_desc_css: CDataDescriptor,
    data_desc_avlb: CDataDescriptor,
    dst_db: str,
    table_sig_stg: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    signals_fac_opt = CSignalsStg(
        tgt_rets=tgt_rets,
        sectors=sectors,
        universe_sector=universe_sector,
        data_desc_pv=data_desc_pv,
        data_desc_optimize=data_desc_optimize,
        data_desc_css=data_desc_css,
        data_desc_avlb=data_desc_avlb,
    )
    signals_fac_opt.set_name("signals_stg")
    mat.add_component(signals_fac_opt)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_sig_stg}"
    create_factor_table(dst_path)
    signals_fac_opt.save_factors(dst_path)
    return 0
