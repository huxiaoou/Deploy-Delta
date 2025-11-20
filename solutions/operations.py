import pandas as pd
import datetime as dt
import os
from qtools_sxzq.qdataviewer import fetch
from qtools_sxzq.qdata import CDataDescriptor
from qtools_sxzq.qcalendar import CCalendar


def convert_real_code_to_contract(real_code: str) -> str:
    if pd.isnull(real_code):
        return None

    contract, exchange = real_code.split("_")
    if exchange in ["SHFE", "INE", "DCE"]:
        return contract.lower()
    elif exchange in ["CZCE"]:
        return contract[:-4] + contract[-3:]
    else:
        raise ValueError(f"Invalid exchange code = {exchange}")


def get_signals_by_date(
    trade_date: str,
    sig_type: str,
    sig_db: CDataDescriptor,
    contract_db: CDataDescriptor,
    save_root_dir: str,
):
    dt_sig = f"{trade_date[0:4]}-{trade_date[4:6]}-{trade_date[6:8]} 15:00:00"
    conds = f"datetime == '{dt_sig}'"
    sig_data: pd.DataFrame = fetch(
        lib=sig_db.db_name,
        table=sig_db.table_name,
        names=["datetime", "code", sig_type],
        conds=conds,
    ).rename(columns={sig_type: "weight"})
    contract_data: pd.DataFrame = fetch(
        lib=contract_db.db_name,
        table=contract_db.table_name,
        names=["datetime", "code", "real_code", "`close`"],
        conds=conds,
    )
    data = pd.merge(left=sig_data, right=contract_data, how="left", on=["datetime", "code"])
    data["contract"] = data["real_code"].map(convert_real_code_to_contract)
    print(data)

    miss_data = data[data.isnull().any(axis=1)]
    if not miss_data.empty:
        print(f"[WRN] Some data are missing")
        print(miss_data)

    data = data[~data.isnull().any(axis=1)]
    if not os.path.exists(d := os.path.join(save_root_dir, "signals", trade_date[0:4], trade_date[4:6])):
        os.makedirs(d)
    data_file = f"signals_sig-date_{trade_date}_{sig_type}.csv"
    data_path = os.path.join(d, data_file)
    data.to_csv(data_path, index=False, float_format="%.8f")
    print(f"[INF] file is saved in {data_path}")
    return 0


def get_signals(
    bgn: str,
    end: str,
    calendar: CCalendar,
    sig_type: str,
    sig_db: CDataDescriptor,
    contract_db: CDataDescriptor,
    save_root_dir: str,
):
    stp = (dt.datetime.strptime(end, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
    dates = calendar.get_iter_list(bgn, stp)
    for trade_date in dates:
        get_signals_by_date(
            trade_date=trade_date,
            sig_type=sig_type,
            sig_db=sig_db,
            contract_db=contract_db,
            save_root_dir=save_root_dir,
        )
