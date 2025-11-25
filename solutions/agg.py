import pandas as pd
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgFactors


def process_by_day(raw_data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """

    :param raw_data: index = code_name, like "AG9999_SHFE".
                columns = ["avlb", "sector", "f1", "f2", ...]
    :param factors:
    :return: nor_data, a pd.Dataframe, index = code_name,
                columns = ["avlb", "sector", "f1", "f2", ...]
    """

    avlb_data = raw_data.query("avlb > 0")
    agg_data: pd.DataFrame = avlb_data.groupby(by="sector").apply(
        lambda z: z["amt"] @ z[factors] / z["amt"].sum()
    )  # type:ignore
    return agg_data


class CFactorsAgg(SignalStrategy):
    def __init__(
        self,
        cfg_factors: CCfgFactors,
        data_desc_avlb: CDataDescriptor,
        data_desc_fac_nrm: CDataDescriptor,
        data_desc_pv: CDataDescriptor,
        universe_sector: dict[str, str],
    ):
        self.cfg_factors: CCfgFactors
        self.data_desc_avlb: CDataDescriptor
        self.data_desc_fac_nrm: CDataDescriptor
        self.data_desc_pv: CDataDescriptor
        self.sectors: list[str]
        self.universe_sector: dict[str, str]
        super().__init__(
            cfg_factors,
            data_desc_avlb,
            data_desc_fac_nrm,
            data_desc_pv,
            universe_sector,
        )

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.subscribe_data("fac_nrm", self.data_desc_fac_nrm.to_args())
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.create_factor_table(self.cfg_factors.to_list())

    def on_clock(self):
        factors = self.cfg_factors.to_list()
        nrm_data = {
            "avlb": self.avlb.get_dict("avlb"),
            "amt": self.pv.get_dict("amt_major"),
            "sector": self.universe_sector,
        }
        for factor in factors:
            nrm_data[factor] = self.fac_nrm.get_dict(factor)
        nrm_data = pd.DataFrame(nrm_data)
        agg_data = process_by_day(nrm_data, factors=factors)
        sorted_data = agg_data.loc[self.codes]
        for factor in factors:
            self.update_factor(factor, sorted_data[factor].to_numpy())


def main_process_factors_agg(
    span: tuple[str, str],
    sectors: list[str],
    cfg_factors: CCfgFactors,
    data_desc_avlb: CDataDescriptor,
    data_desc_fac_nrm: CDataDescriptor,
    data_desc_pv: CDataDescriptor,
    universe_sector: dict[str, str],
    dst_db: str,
    table_fac_agg: str,
):
    cfg = {
        "span": span,
        "codes": sectors,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    factors_agg = CFactorsAgg(
        cfg_factors=cfg_factors,
        data_desc_avlb=data_desc_avlb,
        data_desc_fac_nrm=data_desc_fac_nrm,
        data_desc_pv=data_desc_pv,
        universe_sector=universe_sector,
    )
    factors_agg.set_name("factors_agg")
    mat.add_component(factors_agg)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_fac_agg}"
    create_factor_table(dst_path)
    factors_agg.save_factors(dst_path)
    return 0
