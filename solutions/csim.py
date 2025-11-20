import numpy as np
import pandas as pd
import os
from transmatrix import Strategy, Evaluator, Matrix
from qtools_sxzq.qwidgets import SFY, check_and_mkdir
from qtools_sxzq.qdata import CDataDescriptor, CMarketDescriptor
from szst_dlz.evaluator import SimulationEvaluator
from szst_dlz.pdf_report_generator import PDFReportGenerator
from typedef import TUniverse


class EvaluatorCrossSection(Evaluator):
    def __init__(self, project_data_dir: str, universe: TUniverse, save_id: str, vid: str):
        super().__init__(project_data_dir, universe, save_id, vid)
        self.perf: pd.DataFrame = pd.DataFrame()
        self.perf_sector: dict[str, pd.DataFrame] = {}
        self.universe: TUniverse = universe

    def critic(self):
        # use the following codes to get more detailed information
        # ------------
        # daily_stats = self.get_daily_stats()
        # print(daily_stats)
        # ------------

        ini_cash = self.strategy.ini_cash
        perf = self.get_pnl()
        perf["每日损益"] = perf["pnl"]
        tot_pnl = perf["每日损益"].cumsum() + ini_cash
        perf["策略净值"] = tot_pnl / ini_cash
        perf["日收益率"] = perf["策略净值"].pct_change().fillna(0)
        self.perf = perf

        daily_stats = self.get_daily_stats().reset_index()
        daily_stats["sector"] = daily_stats["code"].map(lambda z: self.universe[z].sectorL1)
        sector_pnl = pd.pivot_table(data=daily_stats, index="datetime", columns="sector", values="cur_pnl", aggfunc=sum)
        sector_ret = sector_pnl.div(tot_pnl, axis=0)
        self.perf_sector["pnl"] = sector_pnl
        self.perf_sector["ret"] = sector_ret

    def show(self):
        # pd.set_option("display.unicode.east_asian_width", True)
        # print(self.perf)

        mu = self.perf["日收益率"].mean()
        sd = self.perf["日收益率"].std()
        ar = mu * 250
        av = sd * np.sqrt(250)
        sharpe = ar / av
        mdd = (1 - self.perf["策略净值"] / self.perf["策略净值"].cummax()).max()
        calmar = ar / mdd
        hpr = self.perf["策略净值"].iloc[-1] - 1

        print(f"Hold Period Return: {hpr:.4f}")
        print(f"Annual Return     : {ar:.4f}")
        print(f"Annual Volatility : {av:.4f}")
        print(f"Maximum drawdown  : {mdd:.4f}")
        print(f"Sharpe ratio      : {sharpe:.4f}")
        print(f"Calmar ratio      : {calmar:.4f}")

        check_and_mkdir(dst_dir := os.path.join(self.project_data_dir, "perfs"))
        self.perf.to_csv(
            os.path.join(dst_dir, f"perf_{self.save_id}.{self.vid}.csv"),
            float_format="%.8f",
        )
        self.perf_sector["pnl"].to_csv(
            os.path.join(dst_dir, f"perf_sector_pnl.{self.save_id}.{self.vid}.csv"),
            float_format="%.2f",
            index_label="trade_date",
        )
        self.perf_sector["ret"].to_csv(
            os.path.join(dst_dir, f"perf_sector_ret.{self.save_id}.{self.vid}.csv"),
            float_format="%.8f",
            index_label="trade_date",
        )


"""
------------------
--- Strategies ---
------------------
"""


class StrategyOperator(Strategy):
    def __init__(self, *args, oi_cap_ratio: float):
        super().__init__(*args, oi_cap_ratio)
        self.prev_equity: float = 0.0
        self.oi_cap_ratio: float = oi_cap_ratio

    def init(self):
        self.subscribe_data("signals", self.data_desc_sig.to_args())
        self.subscribe_data("market", self.data_desc_pv.to_args())
        self.add_scheduler(milestones=["15:00:00"], handler=self.rebalance)
        self.add_scheduler(milestones=["15:05:00"], handler=self.daily_check)

    def execute(self, code: str, prev_qty: int, target_qty: int, price: float):
        if target_qty > 0:
            if prev_qty > target_qty:
                self.sell(price, volume=prev_qty - target_qty, offset="close", code=code, market="future")
            elif prev_qty == target_qty:
                pass
            elif 0 <= prev_qty < target_qty:
                self.buy(price, volume=target_qty - prev_qty, offset="open", code=code, market="future")
            else:  # prev_qty < 0
                self.buy(price, volume=-prev_qty, offset="close", code=code, market="future")
                self.buy(price, volume=target_qty, offset="open", code=code, market="future")
        elif target_qty < 0:
            if prev_qty < target_qty:
                self.buy(price, volume=target_qty - prev_qty, offset="close", code=code, market="future")
            elif prev_qty == target_qty:
                pass
            elif target_qty < prev_qty <= 0:
                self.sell(price, volume=prev_qty - target_qty, offset="open", code=code, market="future")
            else:  # prev_qty > 0
                self.sell(price, volume=prev_qty, offset="close", code=code, market="future")
                self.sell(price, volume=-target_qty, offset="open", code=code, market="future")
        else:  # target_qty == 0:
            if prev_qty > 0:
                self.sell(price, volume=prev_qty, offset="close", code=code, market="future")
            elif prev_qty < 0:
                self.buy(price, volume=-prev_qty, offset="close", code=code, market="future")
        return 0

    def sig_to_tgt_qty(self, code: str, sig: str, balance: float, allocate_ratio: float) -> int:
        """

        :param code: code-like "AU9999_SHFE"
        :param sig: name of signals, usually like "opn", "cls", "omega"
        :param balance: total amount
        :param allocate_ratio:
        :return: int: target position quantity of code

        """

        mult = self.market.get_code(code, fields="multiplier_major")
        oi = self.market.get_code(code, fields="open_interest_major")
        if np.isnan(oi):
            oi = 0
        qty_cap = int(oi * self.oi_cap_ratio * allocate_ratio)

        # use prev close price and signal to decide the quantity
        prc_ary = self.market.get_window_code(code, 2, fields="close_major")
        prc = np.nan if len(prc_ary) < 2 else prc_ary[-2]
        wgt_ary = self.signals.get_window_code(code, 2, fields=sig)
        wgt = 0 if len(wgt_ary) < 2 else wgt_ary[-2]

        qty_raw = balance * allocate_ratio * wgt / prc / mult
        qty = 0 if np.isnan(qty_raw) else int(np.round(qty_raw))
        qty = min(qty, qty_cap)
        return qty

    def rebalance(self):
        raise NotImplementedError()

    def daily_check(self):
        self.prev_equity = self.account.get_equity("close_major")


class StrategyCrossSection(StrategyOperator):
    def __init__(
        self,
        sig: str,
        exe_price: str,
        data_desc_sig: CDataDescriptor,
        data_desc_pv: CDataDescriptor,
        oi_cap_ratio: float,
    ):
        super().__init__(sig, exe_price, data_desc_sig, data_desc_pv, oi_cap_ratio=oi_cap_ratio)

    def rebalance(self):
        for code in self.codes:
            code_pre_qty = self.account.get_netpos(code)
            code_tgt_qty = self.sig_to_tgt_qty(code, sig=self.sig, balance=self.prev_equity, allocate_ratio=1.0)
            price = self.market.get_code(code, fields=self.exe_price)
            self.execute(code, code_pre_qty, code_tgt_qty, price)


def main_process_sim_cmplx(
    span: tuple[str, str],
    codes: list[str],
    sig: str,
    data_desc_sig: CDataDescriptor,
    exe_price: str,
    oi_cap_ratio: float,
    data_desc_pv: CDataDescriptor,
    mkt_desc_fut: CMarketDescriptor,
    universe: TUniverse,
    project_data_dir: str,
    vid: str,
    using_sxzq_dlz: bool,
):
    cfg = {
        "progress_bar": True,
        "span": span,
        "codes": codes,
        "market": {"future": mkt_desc_fut.to_dict()},
        "cache_data": False,
    }

    # --- run
    mat = Matrix(cfg)
    sim_strategy = StrategyCrossSection(
        sig=sig,
        exe_price=exe_price,
        data_desc_sig=data_desc_sig,
        data_desc_pv=data_desc_pv,
        oi_cap_ratio=oi_cap_ratio,
    )
    sim_strategy.set_name("sim_strategy")
    mat.add_component(sim_strategy)

    evaluator0 = EvaluatorCrossSection(
        project_data_dir=project_data_dir,
        universe=universe,
        save_id=f"{sig}-{exe_price}",
        vid=vid,
    )
    evaluator0.set_name("evaluator0")
    mat.add_component(evaluator0)

    eval_dir = os.path.join(project_data_dir, f"eval_{sig}_{vid}")
    if using_sxzq_dlz:
        evaluator1 = SimulationEvaluator(
            bench={"南华商品指数": ["basic", "macro_data", "MACRO_9999", "close", 0]},
            sector=True,
            save_dir=eval_dir,
            sector_grp=2,
        )
        evaluator1.set_name("evaluator1")
        mat.add_component(evaluator1)

    mat.init()
    mat.run()
    mat.eval()

    if using_sxzq_dlz:
        title = f"report_{sig}.{vid}"
        report = PDFReportGenerator(
            title,
            evaluator_name="evaluator1",
            input_data_dir=eval_dir,
            pdf_save_dir=project_data_dir,
        )
        report.genearate_from_strategy(sim_strategy)
    return 0


"""
------------------------
--- 2 sub strategies ---
------------------------
"""


class StrategyCrossSectionDualSubs(StrategyOperator):
    def __init__(
        self,
        sig_0: str,
        exe_price_0: str,
        sig_1: str,
        exe_price_1: str,
        data_desc_sig: CDataDescriptor,
        data_desc_pv: CDataDescriptor,
        codes: list[str],
        oi_cap_ratio: float,
    ):
        super().__init__(
            sig_0,
            exe_price_0,
            sig_1,
            exe_price_1,
            data_desc_sig,
            data_desc_pv,
            oi_cap_ratio=oi_cap_ratio,
        )
        self.sub_pos_mgr: dict[str, dict[str, int]] = {
            sig_0: {code: 0 for code in codes},
            sig_1: {code: 0 for code in codes},
        }

    @property
    def sigs(self) -> list[str]:
        return [self.sig_0, self.sig_1]

    @property
    def exe_prices(self) -> list[str]:
        return [self.exe_price_0, self.exe_price_1]

    def get_code_qty_frm_mgr(self, code: str) -> int:
        return self.sub_pos_mgr[self.sig_0][code] + self.sub_pos_mgr[self.sig_1][code]

    def get_code_strategy_qty_from_mgr(self, code: str) -> tuple[int, int]:
        return self.sub_pos_mgr[self.sig_0][code], self.sub_pos_mgr[self.sig_1][code]

    def rebalance(self):
        for code in self.codes:
            code_pre_qty = self.account.get_netpos(code)
            if code_pre_qty != self.get_code_qty_frm_mgr(code):
                q0, q1 = self.get_code_strategy_qty_from_mgr(code)
                msg = (
                    f"datetime = {SFY(self.time)}, "
                    f"Code = {SFY(code)}, Actual pos qty in account = {SFY(code_pre_qty)}, while "
                    f"pos qty in sub strategy {SFY(self.sig_0)} = {SFY(q0)}, "
                    f"pos qty in sub strategy {SFY(self.sig_1)} = {SFY(q1)}."
                )
                raise ValueError(msg)

        for sig, exe_price in zip(self.sigs, self.exe_prices):
            for code in self.codes:
                code_tgt_qty = self.sig_to_tgt_qty(code, sig=sig, balance=self.prev_equity, allocate_ratio=0.5)
                price = self.market.get_code(code, fields=exe_price)
                code_pre_qty = self.sub_pos_mgr[sig][code]
                self.execute(code, code_pre_qty, code_tgt_qty, price)
                self.sub_pos_mgr[sig][code] = code_tgt_qty


def main_process_sim_dual_sub(
    span: tuple[str, str],
    codes: list[str],
    sig_0: str,
    exe_price_0: str,
    sig_1: str,
    exe_price_1: str,
    oi_cap_ratio: float,
    data_desc_sig: CDataDescriptor,
    data_desc_pv: CDataDescriptor,
    mkt_desc_fut: CMarketDescriptor,
    universe: TUniverse,
    project_data_dir: str,
    vid: str,
    using_sxzq_dlz: bool,
):
    cfg = {
        "progress_bar": True,
        "span": span,
        "codes": codes,
        "market": {"future": mkt_desc_fut.to_dict()},
        "cache_data": False,
    }

    # --- run
    mat = Matrix(cfg)
    sim_strategy = StrategyCrossSectionDualSubs(
        sig_0=sig_0,
        exe_price_0=exe_price_0,
        sig_1=sig_1,
        exe_price_1=exe_price_1,
        oi_cap_ratio=oi_cap_ratio,
        data_desc_sig=data_desc_sig,
        data_desc_pv=data_desc_pv,
        codes=codes,
    )
    sim_strategy.set_name("sim_strategy")
    mat.add_component(sim_strategy)

    evaluator0 = EvaluatorCrossSection(
        project_data_dir=project_data_dir,
        universe=universe,
        save_id="dualSubs",
        vid=vid,
    )
    evaluator0.set_name("evaluator0")
    mat.add_component(evaluator0)

    eval_dir = os.path.join(project_data_dir, f"eval_dualSubs_{vid}")
    if using_sxzq_dlz:
        evaluator1 = SimulationEvaluator(
            bench={"南华商品指数": ["basic", "macro_data", "MACRO_9999", "close", 0]},
            sector=True,
            save_dir=eval_dir,
            sector_grp=2,
        )
        evaluator1.set_name("evaluator1")
        mat.add_component(evaluator1)

    mat.init()
    mat.run()
    mat.eval()

    if using_sxzq_dlz:
        title = f"report_dualSubs.{vid}"
        report = PDFReportGenerator(
            title,
            evaluator_name="evaluator1",
            input_data_dir=eval_dir,
            pdf_save_dir=project_data_dir,
        )
        report.genearate_from_strategy(sim_strategy)
    return 0
