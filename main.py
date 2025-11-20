import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Entry point of this project")
    arg_parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--end", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    arg_parser_subs = arg_parser.add_subparsers(
        title="Position argument to call sub functions",
        dest="switch",
        description="use this position argument to call different functions of this project. "
        "For example: 'python main.py --bgn 20120104 --end 20240826 available'",
        required=True,
    )

    # switch: available
    arg_parser_subs.add_parser(name="available", help="Calculate available universe")

    # switch: css
    arg_parser_subs.add_parser(name="css", help="Calculate cross section stats")

    # swithc: icov
    arg_parser_subs.add_parser(name="icov", help="Calculate instruments covariance")

    # swithc: srets
    arg_parser_subs.add_parser(name="srets", help="Calculate sector returns")

    # switch: factors
    arg_parser_sub = arg_parser_subs.add_parser(name="factors", help="Calculate factors")
    arg_parser_sub.add_argument("--type", type=str, choices=("raw", "nrm"))

    # switch: signals
    arg_parser_sub = arg_parser_subs.add_parser(name="signals", help="generate signals")
    arg_parser_sub.add_argument("--type", type=str, choices=("fac", "stg"))

    # switch: simulations
    arg_parser_sub = arg_parser_subs.add_parser(name="simulations", help="do simulations")
    arg_parser_sub.add_argument("--type", type=str, choices=("fac", "stg", "stg2"))
    arg_parser_sub.add_argument("--omit", default=False, action="store_true")

    # switch: optimize
    arg_parser_subs.add_parser(name="optimize", help="Optimize weights for factors")

    # switch: operations
    arg_parser_sub = arg_parser_subs.add_parser(name="operations", help="do operations")
    arg_parser_sub.add_argument("--type", type=str, choices=("opn", "cls"), required=True)

    return arg_parser.parse_args()


if __name__ == "__main__":
    from config import (
        cfg,
        cfg_tables,
        cfg_dbs,
        data_desc_preprocess,
        data_desc_dominant,
        data_desc_pv,
        data_desc_pv1m,
        data_desc_cpv,
        data_desc_avlb,
        data_desc_css,
        data_desc_icov,
        data_desc_fac_raw,
        data_desc_fac_nrm,
        data_desc_sig_fac,
        data_desc_sig_stg,
        data_desc_sim_fac,
        data_desc_optimize,
    )

    args = parse_args()
    span: tuple[str, str] = (args.bgn, args.end)
    codes: list[str] = cfg.codes

    if args.switch == "available":
        from solutions.avlb import main_process_avlb

        data_desc_preprocess.lag = cfg.avlb.lag
        main_process_avlb(
            span=span,
            codes=codes,
            cfg_avlb=cfg.avlb,
            data_desc_pv=data_desc_preprocess,
            dst_db=cfg_dbs.user,
            table_avlb=cfg_tables.avlb,
        )
    elif args.switch == "css":
        from solutions.css import main_process_css

        data_desc_preprocess.lag, data_desc_avlb.lag = 1, 1
        main_process_css(
            span=span,
            codes=codes,
            cfg_css=cfg.css,
            data_desc_pv=data_desc_preprocess,
            data_desc_avlb=data_desc_avlb,
            dst_db=cfg_dbs.user,
            table_css=cfg_tables.css,
        )
    elif args.switch == "srets":
        from solutions.srets import main_process_srets
        from config import universe_sector

        data_desc_preprocess.lag, data_desc_avlb.lag = 1, 1
        main_process_srets(
            span=span,
            codes=codes,
            universe_sector=universe_sector,
            data_desc_pv=data_desc_preprocess,
            data_desc_avlb=data_desc_avlb,
            dst_db=cfg_dbs.user,
            table_srets=cfg_tables.srets,
            project_data_dir=cfg.project_data_dir,
        )
    elif args.switch == "icov":
        from solutions.icov import main_process_icov

        data_desc_preprocess.lag = 240
        main_process_icov(
            span=span,
            codes=codes,
            cfg_icov=cfg.icov,
            data_desc_pv=data_desc_preprocess,
            dst_db=cfg_dbs.user,
            table_icov=cfg_tables.icov,
        )
    elif args.switch == "factors":
        if args.type == "raw":
            from solutions.factors import main_process_factors_raw

            data_desc_preprocess.lag = data_desc_pv1m.lag = cfg.factors.lag
            main_process_factors_raw(
                span=span,
                codes=codes,
                cfg_factors=cfg.factors,
                data_desc_pv=data_desc_preprocess,
                data_desc_pv1m=data_desc_pv1m,
                dst_db=cfg_dbs.user,
                table_fac_raw=cfg_tables.fac_raw,
            )
        elif args.type == "nrm":
            from solutions.nrm import main_process_factors_nrm
            from config import universe_sector

            main_process_factors_nrm(
                span=span,
                codes=codes,
                cfg_factors=cfg.factors,
                data_desc_avlb=data_desc_avlb,
                data_desc_fac_raw=data_desc_fac_raw,
                universe_sector=universe_sector,
                dst_db=cfg_dbs.user,
                table_fac_neu=cfg_tables.fac_nrm,
            )
    elif args.switch == "signals":
        if args.type == "fac":
            from solutions.signals import main_process_signals_fac

            main_process_signals_fac(
                span=span,
                codes=codes,
                cfg_factors=cfg.factors,
                data_desc_avlb=data_desc_avlb,
                data_desc_fac_nrm=data_desc_fac_nrm,
                dst_db=cfg_dbs.user,
                table_sig_fac=cfg_tables.sig_fac,
            )
        elif args.type == "stg":
            from solutions.signals import main_process_signals_stg

            main_process_signals_stg(
                span=span,
                codes=codes,
                tgt_rets=cfg.tgt_rets,
                cfg_factors=cfg.factors,
                cfg_qsim=cfg.qsim,
                data_desc_sig_fac=data_desc_sig_fac,
                data_desc_css=data_desc_css,
                data_desc_icov=data_desc_icov,
                data_desc_optimize=data_desc_optimize,
                data_desc_avlb=data_desc_avlb,
                dst_db=cfg_dbs.user,
                table_sig_stg=cfg_tables.sig_stg,
            )
    elif args.switch == "simulations":
        if args.type == "fac":
            from solutions.qsim import CSimQuick

            sim_quick = CSimQuick(
                codes=codes,
                cfg_factors=cfg.factors,
                data_desc_pv=data_desc_preprocess,
                data_desc_sig_fac=data_desc_sig_fac,
                tgt_rets=cfg.tgt_rets,
                cost_rate=cfg.cost_rate_sub,
                dst_db=cfg_dbs.user,
                table_sim_fac=cfg_tables.sim_fac,
                project_data_dir=cfg.project_data_dir,
                vid=cfg.vid,
            )
            sim_quick.main(span=span, ret_win=cfg.qsim.win)
        elif args.type == "stg":
            import os
            from solutions.csim import main_process_sim_cmplx
            from solutions.csim import main_process_sim_dual_sub
            from solutions.eval import CMultiEvaluator
            from config import mkt_desc_fut, universe

            # mkt_desc_fut.settle_price_field = "close"
            for tgt_ret in cfg.tgt_rets:
                exe_price = "open_major" if tgt_ret == "opn" else "close_major"
                main_process_sim_cmplx(
                    span=span,
                    codes=codes,
                    sig=tgt_ret,
                    data_desc_sig=data_desc_sig_stg,
                    exe_price=exe_price,
                    oi_cap_ratio=cfg.avlb.oi_cap_ratio,
                    data_desc_pv=data_desc_preprocess,
                    mkt_desc_fut=mkt_desc_fut,
                    project_data_dir=cfg.project_data_dir,
                    universe=universe,
                    vid=cfg.vid,
                    using_sxzq_dlz=not args.omit,
                )
            sig_0, sig_1 = "opn", "cls"
            exe_price_0, exe_price_1 = "open_major", "close_major"
            main_process_sim_dual_sub(
                span=span,
                codes=codes,
                sig_0=sig_0,
                exe_price_0=exe_price_0,
                sig_1=sig_1,
                exe_price_1=exe_price_1,
                oi_cap_ratio=cfg.avlb.oi_cap_ratio,
                data_desc_sig=data_desc_sig_stg,
                data_desc_pv=data_desc_preprocess,
                mkt_desc_fut=mkt_desc_fut,
                project_data_dir=cfg.project_data_dir,
                universe=universe,
                vid=cfg.vid,
                using_sxzq_dlz=not args.omit,
            )
            mulit_evaluator = CMultiEvaluator(
                perf_paths=[
                    os.path.join(cfg.project_data_dir, "perfs", f"perf_{sig_0}-{exe_price_0}.{cfg.vid}.csv"),
                    os.path.join(cfg.project_data_dir, "perfs", f"perf_{sig_1}-{exe_price_1}.{cfg.vid}.csv"),
                    os.path.join(cfg.project_data_dir, "perfs", f"perf_dualSubs.{cfg.vid}.csv"),
                ],
                ret_lbl="日收益率",
                date_lbl="date",
                short_ids=["open", "close", "dual"],
                by_year_ids=["open"],
                project_data_dir=cfg.project_data_dir,
                src_id="csim",
                vid=cfg.vid,
            )
            mulit_evaluator.main()
        elif args.type == "stg2":
            import os
            from qtools_sxzq.qsimulation import TExePriceType, CSignal, CMgrMktData, CMgrMajContract, CSimulation
            from qtools_sxzq.qcalendar import CCalendar
            from solutions.eval import CMultiEvaluator

            calendar = CCalendar(calendar_path=cfg.path_calendar)

            bgn_date, stp_date = span[0], calendar.get_next_date(span[1], 1)
            mgr_maj = CMgrMajContract(universe=cfg.codes, dominant=data_desc_dominant)
            mgr_md = CMgrMktData(fmd=data_desc_cpv)
            for tgt_ret in cfg.tgt_rets:
                exe_price = TExePriceType.OPEN if tgt_ret == "opn" else TExePriceType.CLOSE
                signal = CSignal(sid=tgt_ret, signal_db=data_desc_sig_stg)
                sim = CSimulation(
                    signal=signal,
                    init_cash=cfg.init_cash,
                    cost_rate=cfg.cost_rate_pri,
                    exe_price_type=exe_price,
                    mgr_maj_contract=mgr_maj,
                    mgr_mkt_data=mgr_md,
                    sim_save_dir=os.path.join(cfg.project_data_dir, "perfs"),
                    vid=cfg.vid,
                )
                sim.main(bgn_date=bgn_date, stp_date=stp_date, calendar=calendar)

            sig_0, sig_1 = "opn", "cls"
            exe_price_0, exe_price_1 = "open", "close"
            mulit_evaluator = CMultiEvaluator(
                perf_paths=[
                    os.path.join(cfg.project_data_dir, "perfs", f"hsim_{sig_0}-{exe_price_0}.{cfg.vid}.csv"),
                    os.path.join(cfg.project_data_dir, "perfs", f"hsim_{sig_1}-{exe_price_1}.{cfg.vid}.csv"),
                ],
                ret_lbl="ret",
                date_lbl="trade_date",
                short_ids=["open", "close"],
                by_year_ids=["open"],
                project_data_dir=cfg.project_data_dir,
                src_id="hsim",
                vid=cfg.vid,
            )
            mulit_evaluator.main()

    elif args.switch == "optimize":
        from solutions.optimize import main_process_optimize_fac_wgt

        main_process_optimize_fac_wgt(
            span=span,
            codes=cfg.sim_codes_fac,  # codes here are different from usual. they have format like "mtm-opn"
            cfg_factors=cfg.factors,
            tgt_rets=cfg.tgt_rets,
            cfg_optimizer=cfg.optimizer,
            data_desc_sim=data_desc_sim_fac,
            dst_db=cfg_dbs.user,
            table_optimize=cfg_tables.optimize,
        )
    elif args.switch == "operations":
        from solutions.operations import get_signals
        from qtools_sxzq.qcalendar import CCalendar

        calendar = CCalendar(calendar_path=cfg.path_calendar)

        get_signals(
            bgn=args.bgn,
            end=args.end,
            calendar=calendar,
            sig_type=args.type,
            sig_db=data_desc_sig_stg,
            contract_db=data_desc_pv,
            save_root_dir=cfg.project_data_dir,
        )
    else:
        raise ValueError(f"Invalid argument 'switch' = {args.switch}")
