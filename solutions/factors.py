import pandas as pd
from tqdm import tqdm
from typedef import CCfgFactors
from transmatrix import SignalMatrix
from transmatrix.strategy import SignalStrategy
from transmatrix.data_api import create_factor_table
from transmatrix.data_api import NdarrayData, create_data_view, DataView3d
from qtools_sxzq.qdata import CDataDescriptor
from typedef_factor import TInterFactorData
from solutions.math_tools import robust_div


pd.set_option("display.unicode.east_asian_width", True)


class CFactorsRaw(SignalStrategy):
    def __init__(
        self,
        cfg_factors: CCfgFactors,
        data_desc_pv: CDataDescriptor,
        data_desc_pv1m: CDataDescriptor,
    ):
        super().__init__(
            cfg_factors,
            data_desc_pv,
            data_desc_pv1m,
        )
        self.cfg_factors = cfg_factors
        self.pre_trans_factors: DataView3d = None

    def init(self):
        self.add_clock(milestones="15:00:00")
        self.subscribe_data("pv", self.data_desc_pv.to_args())
        self.subscribe_data("pv1m", self.data_desc_pv1m.to_args())
        self.create_factor_table(self.cfg_factors.to_list())

    @property
    def codes_minor(self) -> list[str]:
        return [z.replace("9999", "8888") for z in self.codes]

    @staticmethod
    def update_header_from_factor(header: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(columns=header.columns, index=header.index)
        df.update(factor)
        return df

    def pre_transform(self):
        pv_header = self.pv.to_dataframe()["open_major"]
        pv = self.pv.data.to_dataframe(col="field")
        pv1m = self.pv1m.data.to_dataframe(col="field")
        rename_mapper_1m = {
            "trade_day": "trade_day",
            "turnover": "turnover",
            "volume": "vol",
            "open_interest": "oi",
            "pre_close_ret": "ret",
            "high": "high",
            "low": "low",
        }
        factor_data: TInterFactorData = {}
        for code in tqdm(self.codes, desc="... calculating pre_trans_factors"):
            instru_data_1m = pv1m[code][list(rename_mapper_1m)]
            instru_data_1m = instru_data_1m.rename(columns=rename_mapper_1m).dropna(axis=0, how="all")
            instru_data_pv = pv[code]
            instru_data_1d = pd.DataFrame(
                {
                    "cls_n": instru_data_pv["close_major"],
                    "cls_d": instru_data_pv["close_minor"],
                    "ticker_n": instru_data_pv["code_major"],
                    "ticker_d": instru_data_pv["code_minor"],
                }
            )
            for factor, (_, alg) in self.cfg_factors.mgr.items():
                alg.update_factor_data_in_pre_trans_form(
                    code=code,
                    factor_data=factor_data,
                    instru_data_1d=instru_data_1d,
                    instru_data_1m=instru_data_1m,
                )

        for k, v in factor_data.items():
            factor_data[k] = self.update_header_from_factor(pv_header, pd.DataFrame(v))

        # --- save to DataView3d
        if factor_data:
            self.pre_trans_factors = create_data_view(NdarrayData.from_dataframes(factor_data))
            self.pre_trans_factors.align_with(self.pv)
        else:
            self.pre_trans_factors = None

    def on_clock(self):
        safe_lag = self.cfg_factors.lag
        pv = dict(
            ret=self.pv.get_window_df("pre_cls_ret_major", safe_lag)[self.codes],
            ret_minor=self.pv.get_window_df("pre_cls_ret_minor", safe_lag)[self.codes],
            close=self.pv.get_window_df("close_major", safe_lag)[self.codes],
            turnover=self.pv.get_window_df("amt_major", safe_lag)[self.codes],
            oi=self.pv.get_window_df("open_interest_major", safe_lag)[self.codes],
            vol=self.pv.get_window_df("volume_major", safe_lag)[self.codes],
            basis_rate=self.pv.get_window_df("basis_rate", safe_lag)[self.codes],
            stock=self.pv.get_window_df("stock", safe_lag)[self.codes],
        )
        pv["prc"] = (pv["ret"] + 1).cumprod()
        pv["aver_oi"] = pv["oi"].rolling(window=2).mean()
        pv["to_rate"] = robust_div(pv["vol"], pv["aver_oi"])
        intermediary = (
            {k: self.pre_trans_factors.get_window_df(k, safe_lag)[self.codes] for k in self.pre_trans_factors.fields}
            if self.pre_trans_factors
            else {}
        )
        other_kwargs = {**pv, **intermediary}
        for factor, (_, alg) in self.cfg_factors.mgr.items():
            fac_val = alg.cal_factor(**other_kwargs)
            self.update_factor(factor, fac_val)


def main_process_factors_raw(
    span: tuple[str, str],
    codes: list[str],
    cfg_factors: CCfgFactors,
    data_desc_pv: CDataDescriptor,
    data_desc_pv1m: CDataDescriptor,
    dst_db: str,
    table_fac_raw: str,
):
    cfg = {
        "span": span,
        "codes": codes,
        "cache_data": False,
        "progress_bar": True,
    }

    # --- run
    mat = SignalMatrix(cfg)
    factors_raw = CFactorsRaw(
        cfg_factors=cfg_factors,
        data_desc_pv=data_desc_pv,
        data_desc_pv1m=data_desc_pv1m,
    )
    factors_raw.set_name("factors_raw")
    mat.add_component(factors_raw)
    mat.init()
    mat.run()

    # --- save
    dst_path = f"{dst_db}.{table_fac_raw}"
    create_factor_table(dst_path)
    factors_raw.save_factors(dst_path)
    return 0
