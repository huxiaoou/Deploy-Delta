import numpy as np
import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef_factor import CCfgFactors
from typedef import CCfgQSim
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
            self.update_factor(factor, sorted_data["weight"].values)


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
        cfg_factors: CCfgFactors,
        cfg_qsim: CCfgQSim,
        data_desc_sig_fac: CDataDescriptor,
        data_desc_optimize: CDataDescriptor,
        data_desc_css: CDataDescriptor,
        data_desc_icov: CDataDescriptor,
        data_desc_avlb: CDataDescriptor,
    ):
        super().__init__(
            tgt_rets,
            cfg_factors,
            cfg_qsim,
            data_desc_sig_fac,
            data_desc_optimize,
            data_desc_css,
            data_desc_icov,
            data_desc_avlb,
        )

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("sig_fac", self.data_desc_sig_fac.to_args())
        self.subscribe_data("optimize", self.data_desc_optimize.to_args())
        self.subscribe_data("css", self.data_desc_css.to_args())
        self.subscribe_data("icov", self.data_desc_icov.to_args())
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.create_factor_table(self.tgt_rets)

    def get_icov(self) -> pd.DataFrame:
        icov = pd.concat(self.icov.query(self.time, periods=1))
        icov = icov.reset_index(level=1, drop=True)
        icov.index = icov.index.map(lambda _: _.upper())
        return icov

    def get_factor_ma(self, factor: str, norm: bool = True) -> pd.Series:
        data = self.sig_fac.get_window_df(factor, self.cfg_qsim.win).fillna(0)
        m: pd.Series = data.mean()
        if norm and (abs_sum := m.abs().sum()) > 0:
            m = m / abs_sum
        return m

    def adj_cov(self, raw_wgt: pd.Series, icov: pd.DataFrame) -> pd.Series:
        sort_wgt = raw_wgt.sort_values(ascending=False)
        w0 = sort_wgt.to_numpy()
        k, k0 = len(w0), len(w0[w0 >= 0])
        k1 = k - k0
        top_list = sort_wgt.head(k0).index.tolist()
        btm_list = sort_wgt.tail(k1).index.tolist()
        cov_top = icov.loc[top_list, top_list]
        cov_btm = icov.loc[btm_list, btm_list]
        w_top, w_btm = w0[0:k0], w0[-k1:]
        var_top, var_btm = w_top @ cov_top @ w_top, w_btm @ cov_btm @ w_btm
        top_btm_ratio = np.sqrt(var_top / var_btm)
        # top_btm_ratio = 1.0
        w0[-k1:] = w0[-k1:] * top_btm_ratio
        w0 = w0 / np.sum(np.abs(w0))
        res = pd.Series(data=w0, index=sort_wgt.index)
        return res[self.codes]

    def on_clock(self):
        tot_wgt: float = self.css.get_dict("val")["TOTWGT"]
        icov = self.get_icov()
        for tgt_ret in self.tgt_rets:
            wgt = pd.Series(self.optimize.get_dict(tgt_ret))[self.cfg_factors.to_list()]
            fac_data = {}
            for factor in self.cfg_factors.to_list():
                fac_data[factor] = self.get_factor_ma(factor=factor)
            fac_data = pd.DataFrame(fac_data)
            raw_wgt = (fac_data @ wgt)[self.codes]
            adj_wgt = self.adj_cov(raw_wgt=raw_wgt, icov=icov)
            opt_wgt = adj_wgt * tot_wgt
            self.update_factor(tgt_ret, opt_wgt.to_numpy())


def main_process_signals_stg(
    span: tuple[str, str],
    codes: list[str],
    tgt_rets: list[str],
    cfg_factors: CCfgFactors,
    cfg_qsim: CCfgQSim,
    data_desc_sig_fac: CDataDescriptor,
    data_desc_optimize: CDataDescriptor,
    data_desc_css: CDataDescriptor,
    data_desc_icov: CDataDescriptor,
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
        cfg_factors=cfg_factors,
        cfg_qsim=cfg_qsim,
        data_desc_sig_fac=data_desc_sig_fac,
        data_desc_css=data_desc_css,
        data_desc_icov=data_desc_icov,
        data_desc_optimize=data_desc_optimize,
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
