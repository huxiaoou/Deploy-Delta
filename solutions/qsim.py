import os
import pandas as pd
from itertools import product
from tqdm import tqdm
from qtools_sxzq.qwidgets import check_and_mkdir
from qtools_sxzq.qdataviewer import fetch
from qtools_sxzq.qdata import CDataDescriptor, save_data3d_to_db_with_key_as_code
from qtools_sxzq.qplot import CPlotLines
from typedef import CCfgFactors, CSimArgs


def sim_ret(wgt_data: pd.DataFrame, ret_data: pd.DataFrame, cost_rate: float) -> pd.DataFrame:
    raw_ret = (wgt_data * ret_data).sum(axis=1)
    wgt_data_prev = wgt_data.shift(1).fillna(0)
    wgt_diff = wgt_data - wgt_data_prev
    dlt_wgt = wgt_diff.abs().sum(axis=1)
    cost = dlt_wgt * cost_rate
    net_ret = raw_ret - cost
    sim_data = pd.DataFrame({"raw_ret": raw_ret, "dlt_wgt": dlt_wgt, "cost": cost, "net_ret": net_ret})
    return sim_data


class CSimQuick:
    def __init__(
        self,
        codes: list[str],
        cfg_factors: CCfgFactors,
        tgt_rets: list[str],
        data_desc_pv: CDataDescriptor,
        data_desc_sig_fac: CDataDescriptor,
        cost_rate: float,
        dst_db: str,
        table_sim_fac: str,
        project_data_dir: str,
        vid: str,
    ):
        self.codes = codes
        self.cfg_factors = cfg_factors
        self.data_desc_pv = data_desc_pv
        self.data_desc_sig_fac_ma = data_desc_sig_fac
        self.tgt_rets = tgt_rets
        self.cost_rate = cost_rate
        self.dst_db = dst_db
        self.table_sim_fac_ma = table_sim_fac
        self.project_data_dir = project_data_dir
        self.vid = vid

    def load_ret(self, span: tuple[str, str], ret_win: int) -> dict[str, pd.DataFrame]:
        b, e = span
        bgn, end = f"{b[0:4]}-{b[4:6]}-{b[6:8]}", f"{e[0:4]}-{e[4:6]}-{e[6:8]}"
        data = fetch(
            lib=self.data_desc_pv.db_name,
            table=self.data_desc_pv.table_name,
            names="datetime,code,pre_cls_ret_major,pre_opn_ret_major",
            conds=f"datetime >= '{bgn} 15:00:00' and datetime <= '{end} 15:00:00'",
        )
        opn_ret = pd.pivot_table(data=data, index="datetime", columns="code", values="pre_opn_ret_major")[self.codes]
        cls_ret = pd.pivot_table(data=data, index="datetime", columns="code", values="pre_cls_ret_major")[self.codes]
        opn_ret_adj = opn_ret.dropna(axis=0, how="all").fillna(0).rolling(window=ret_win, min_periods=1).mean()
        cls_ret_adj = cls_ret.dropna(axis=0, how="all").fillna(0).rolling(window=ret_win, min_periods=1).mean()
        return {
            "opn": opn_ret_adj,
            "cls": cls_ret_adj,
        }

    def load_sig(self, span: tuple[str, str]) -> dict[str, pd.DataFrame]:
        b, e = span
        bgn, end = f"{b[0:4]}-{b[4:6]}-{b[6:8]}", f"{e[0:4]}-{e[4:6]}-{e[6:8]}"
        factors = self.cfg_factors.to_list()
        data = fetch(
            lib=self.data_desc_sig_fac_ma.db_name,
            table=self.data_desc_sig_fac_ma.table_name,
            names=["datetime", "code"] + factors,
            conds=f"datetime >= '{bgn} 15:00:00' and datetime <= '{end} 15:00:00'",
        )
        piv_data = pd.pivot_table(data=data, index="datetime", columns="code", values=factors)
        res: dict[str, pd.DataFrame] = {}
        for factor in factors:
            res[factor] = piv_data[factor][self.codes]  # type:ignore
        return res

    def get_net_ret(
        self, tgt_ret: str, sim_args_grp: list[CSimArgs], sim_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        net_ret_data = {}
        for sim_args in sim_args_grp:
            if sim_args.ret == tgt_ret:
                net_ret_data[sim_args.save_id] = sim_data[sim_args.save_id]["net_ret"]
        return pd.DataFrame(net_ret_data)

    def plot_by_tgt_ret(self, net_ret: pd.DataFrame, tgt_ret: str):
        nav_data = (net_ret + 1).cumprod(axis=0)
        nav_data.index = map(lambda z: z.strftime("%Y/%m/%d"), nav_data.index)  # type:ignore
        k = nav_data.shape[1]
        artist = CPlotLines(
            plot_data=nav_data,
            colormap="jet",
            line_style=["-", "-."] * (k // 2),
            line_width=1.2,
        )
        artist.plot()
        artist.set_legend(loc="upper left")
        artist.set_axis_x(xtick_count=50, xtick_label_size=12, xtick_label_rotation=90, xgrid_visible=True)
        artist.set_axis_y(ylim=(0.90, 1.80), ytick_spread=0.05, update_yticklabels=False, ygrid_visible=True)
        check_and_mkdir(dst_dir := os.path.join(self.project_data_dir, "plots"))
        artist.save(
            fig_name=f"sim_fac_ma_{tgt_ret}.{self.vid}",
            fig_save_dir=dst_dir,
            fig_save_type="pdf",
        )
        artist.close()
        return 0

    def main(self, span: tuple[str, str], ret_win: int):
        pv = self.load_ret(span=span, ret_win=ret_win)
        sig = self.load_sig(span=span)
        sim_args_grp = [
            CSimArgs(sig=factor, ret=tgt_ret) for factor, tgt_ret in product(self.cfg_factors.to_list(), self.tgt_rets)
        ]
        sim_data: dict[str, pd.DataFrame] = {}
        for sim_args in tqdm(sim_args_grp):
            ret_data = pv[sim_args.ret]
            sig_data = sig[sim_args.sig]
            if len(ret_data) != len(sig_data):
                raise ValueError(f"length of sig != length of ret")
            # factor data @ "T 15:00:00"
            # cls(opn) return @ "T+w+1 15:00:00" means "cls[T+1] -> cls[T+1+w]"("opn[T+1] -> opn[T+w+1]"), in which w = ret_win.
            # so use shift = 2 to align
            wgt_data = sig_data.shift(1 + ret_win).fillna(0)
            fac_sim_data = sim_ret(wgt_data, ret_data=ret_data, cost_rate=self.cost_rate)
            sim_data[sim_args.save_id] = fac_sim_data

        save_data3d_to_db_with_key_as_code(data_3d=sim_data, db_name=self.dst_db, table_name=self.table_sim_fac_ma)
        for tgt_ret in self.tgt_rets:
            net_ret = self.get_net_ret(tgt_ret=tgt_ret, sim_args_grp=sim_args_grp, sim_data=sim_data)
            self.plot_by_tgt_ret(net_ret=net_ret, tgt_ret=tgt_ret)
        return 0
