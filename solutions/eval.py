import os
import pandas as pd
from qtools_sxzq.qwidgets import check_and_mkdir
from qtools_sxzq.qplot import CPlotLines
from qtools_sxzq.qevaluation import CNAV


def plot_nav(
    nav_data: pd.DataFrame,
    xtick_count_min: int,
    ylim: tuple[float, float],
    ytick_spread: float,
    fig_name: str,
    save_dir: str,
    line_style: list = None,
    line_color: list = None,
    colormap: str = "jet",
):
    artist = CPlotLines(
        plot_data=nav_data,
        line_width=1.2,
        line_style=line_style,
        line_color=line_color,
        colormap=colormap,
    )
    artist.plot()
    artist.set_axis_x(
        xtick_count=min(xtick_count_min, len(nav_data)),
        xtick_label_size=12,
        xtick_label_rotation=90,
        xgrid_visible=True,
    )
    artist.set_axis_y(
        ylim=ylim,
        ytick_spread=ytick_spread,
        update_yticklabels=False,
        ygrid_visible=True,
    )
    check_and_mkdir(save_dir)
    artist.save(
        fig_name=fig_name,
        fig_save_dir=save_dir,
        fig_save_type="pdf",
    )
    artist.close()
    return


class CMultiEvaluator:
    SEP = "=" * 60
    INDICATORS = ["hpr", "retMean", "retStd", "retAnnual", "volAnnual", "sharpe", "calmar", "mdd"]

    def __init__(
        self,
        perf_paths: list[str],
        ret_lbl: str,
        date_lbl: str,
        short_ids: list[str],
        by_year_ids: list[str],
        project_data_dir: str,
        src_id: str,
        vid: str,
    ):
        self.perf_paths = perf_paths
        self.ret_lbl = ret_lbl
        self.date_lbl = date_lbl
        self.short_ids = short_ids
        self.by_year_ids = by_year_ids
        self.src_id = src_id
        self.vid = vid
        self.project_data_dir = project_data_dir

    @property
    def save_id(self) -> str:
        return f"{self.src_id}.{self.vid}"

    def eval_by_year(self, perf: pd.DataFrame, short_id: str):
        summary_year = {}
        for trade_year, trade_year_data in perf.groupby("trade_year"):
            y_ret = trade_year_data[self.ret_lbl]
            y_nav = CNAV(input_srs=y_ret, input_type="RET")
            y_nav.cal_all_indicators(qs=(5, 20, 80, 95))
            summary_year[trade_year] = y_nav.to_dict()
        summary_year = pd.DataFrame.from_dict(summary_year, orient="index")[self.INDICATORS]
        summary_year.to_csv(
            os.path.join(self.project_data_dir, f"summary_year_{short_id}.{self.save_id}.csv"),
            float_format="%.6f",
            index_label="year",
        )
        print(self.SEP)
        print(summary_year)
        return

    def summary_all(self, res: dict[str, dict]):
        summary = pd.DataFrame.from_dict(res, orient="index")[self.INDICATORS]
        summary["score"] = summary["sharpe"] + summary["calmar"]
        summary = summary.sort_values(by="score", ascending=False)
        summary.to_csv(
            os.path.join(self.project_data_dir, f"summary.{self.save_id}.csv"),
            float_format="%.6f",
            index_label="portfolios",
        )
        bst_score, bst_sharpe, bst_calmar = (
            summary["score"].iloc[0],
            summary["sharpe"].iloc[0],
            summary["calmar"].iloc[0],
        )
        print(self.SEP)
        print(f"Best score = {bst_score:.2f}, (sharpe, calmar) = ({bst_sharpe:.2f}, {bst_calmar:.2f})")
        print(summary)

    def save_portfolios_rets(self, portfolios_rets: pd.DataFrame):
        portfolios_rets.to_csv(
            os.path.join(self.project_data_dir, f"rets_ALL.{self.save_id}.csv"),
            float_format="%.8f",
            index_label="trade_date",
        )
        return

    def save_portfolios_corr(self, rets_corr: pd.DataFrame):
        rets_corr.to_csv(
            os.path.join(self.project_data_dir, f"corr_ALL.{self.save_id}.csv"),
            float_format="%.6f",
            index_label="portfolios",
        )
        print(self.SEP)
        print(rets_corr)
        return

    def main(self):
        res: dict[str, dict] = {}
        ret_data = {}
        for perf_path, short_id in zip(self.perf_paths, self.short_ids):
            perf = pd.read_csv(perf_path, dtype={self.date_lbl: str}).set_index(self.date_lbl)
            ret_srs = perf[self.ret_lbl]
            nav = CNAV(input_srs=ret_srs, input_type="RET")
            nav.cal_all_indicators(qs=(5, 20, 80, 95))
            res[short_id] = nav.to_dict()
            if short_id in self.by_year_ids:
                perf["trade_year"] = perf.index.map(lambda x: x[0:4])
                self.eval_by_year(perf=perf, short_id=short_id)
            ret_data[short_id] = ret_srs
        self.summary_all(res=res)

        portfolios_rets = pd.DataFrame(ret_data)
        self.save_portfolios_rets(portfolios_rets)
        rets_corr = portfolios_rets.corr()
        self.save_portfolios_corr(rets_corr=rets_corr)

        # plot-ALL
        nav_data = (portfolios_rets + 1).cumprod()
        plot_nav(
            nav_data=nav_data,
            xtick_count_min=60,
            ylim=(1.0, 1.9),
            ytick_spread=0.10,
            fig_name=f"sim_cmplx.{self.save_id}",
            save_dir=os.path.join(self.project_data_dir, "plots"),
        )

        # plot-SINCE
        latest_ret = portfolios_rets.truncate(before="2025-01-01")
        latest_nav = (latest_ret + 1).cumprod()
        plot_nav(
            nav_data=latest_nav,
            xtick_count_min=20,
            ylim=(0.99, 1.06),
            ytick_spread=0.005,
            fig_name=f"sim_cmplx_since_2025.{self.save_id}",
            save_dir=os.path.join(self.project_data_dir, "plots"),
        )
        return 0
