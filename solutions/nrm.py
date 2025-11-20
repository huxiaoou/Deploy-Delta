from typing import Callable
import pandas as pd
import scipy.stats as sps
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from qtools_sxzq.qdata import CDataDescriptor
from typedef import CCfgFactors


def normalize(data: pd.DataFrame, q: float = 0.995) -> pd.DataFrame:
    k = sps.norm.ppf(q)
    mu = data.mean()
    sd = data.std()
    ub, lb = mu + k * sd, mu - k * sd
    t = data.copy()
    for col in data.columns:
        t[col] = t[col].mask(t[col] > ub[col], other=ub[col])
        t[col] = t[col].mask(t[col] < lb[col], other=lb[col])
    z = (t - t.mean()) / t.std()
    return z


def process_by_day(raw_data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """

    :param raw_data: index = code_name, like "AG9999_SHFE".
                columns = ["avlb", "sector", "f1", "f2", ...]
    :param factors:
    :return: nor_data, a pd.Dataframe, index = code_name,
                columns = ["avlb", "sector", "f1", "f2", ...]
    """

    avlb_data = raw_data.query("avlb > 0")
    fil_data = avlb_data.groupby(by="sector")[factors].apply(lambda z: z.fillna(z.mean()))
    nor_data = normalize(data=fil_data)
    res = pd.merge(
        left=raw_data[["avlb", "sector"]],
        right=nor_data,
        left_index=True,
        right_index=True,
        how="left",
    )
    return res


class CFactorsNrm(SignalStrategy):
    def __init__(
        self,
        cfg_factors: CCfgFactors,
        data_desc_avlb: CDataDescriptor,
        data_desc_fac_raw: CDataDescriptor,
        universe_sector: dict[str, str],
    ):
        super().__init__(
            cfg_factors,
            data_desc_avlb,
            data_desc_fac_raw,
            universe_sector,
        )

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("avlb", self.data_desc_avlb.to_args())
        self.subscribe_data("fac_raw", self.data_desc_fac_raw.to_args())
        self.create_factor_table(self.cfg_factors.to_list())

    def on_clock(self):
        factors = self.cfg_factors.to_list()
        raw_data = {
            "avlb": self.avlb.get_dict("avlb"),
            "sector": self.universe_sector,
        }
        for factor in factors:
            raw_data[factor] = self.fac_raw.get_dict(factor)
        raw_data = pd.DataFrame(raw_data)
        nrm_data = process_by_day(raw_data, factors=factors)
        sorted_data = nrm_data.loc[self.codes]
        for factor in factors:
            self.update_factor(factor, sorted_data[factor].values)


def main_process_factors_nrm(
    span: tuple[str, str],
    codes: list[str],
    cfg_factors: CCfgFactors,
    data_desc_avlb: CDataDescriptor,
    data_desc_fac_raw: CDataDescriptor,
    universe_sector: dict[str, str],
    dst_db: str,
    table_fac_neu: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    factors_neu = CFactorsNrm(
        cfg_factors=cfg_factors,
        data_desc_avlb=data_desc_avlb,
        data_desc_fac_raw=data_desc_fac_raw,
        universe_sector=universe_sector,
    )
    factors_neu.set_name("factors_nrm")
    mat.add_component(factors_neu)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_fac_neu}"
    create_factor_table(dst_path)
    factors_neu.save_factors(dst_path)
    return 0
