import numpy as np
import pandas as pd
from typing import Union, Literal


# --------------- Algs for factors ---------------


def gen_exp_wgt(k: int, rate: float = 0.30) -> np.ndarray:
    k0, d = k // 2, k % 2
    rou = np.power(rate, 1 / (k0 - 1)) if k0 > 1 else 1
    sgn = np.array([1] * k0 + [0] * d + [-1] * k0)
    val = np.power(rou, list(range(k0)) + [k0] * d + list(range(k0 - 1, -1, -1)))
    s = sgn * val
    abs_sum = np.abs(s).sum()
    wgt = (s / abs_sum) if abs_sum > 0 else np.zeros(k)
    return wgt


def weighted_mean(x: Union[pd.Series, pd.DataFrame], wgt: pd.Series = None) -> float:
    if wgt is None:
        return x.mean()
    else:
        w = wgt / wgt.abs().sum()
        return w @ x


def weighted_volatility(x: pd.Series, wgt: pd.Series = None) -> float:
    if wgt is None:
        return x.std()
    else:
        w = wgt / wgt.abs().sum()
        mu = x @ w
        x2 = (x**2) @ w
        return np.sqrt(x2 - mu**2)


def robust_div(
    x: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    nan_val: float = np.nan,
) -> Union[pd.Series, pd.DataFrame]:
    """

    :param x: must have the same shape as y
    :param y:
    :param nan_val:
    :return:
    """

    return (x / y.where(y != 0, np.nan)).fillna(nan_val)  # type:ignore


def robust_ret(
    x: pd.Series,
    y: pd.Series,
    scale: float = 1.0,
    condition: Literal["ne", "ge", "le"] = "ne",
) -> pd.Series:
    """

    :param x: must have the same length as y
    :param y:
    :param scale: return scale
    :param condition:
    :return:
    """
    if condition == "ne":
        return (x / y.where(y != 0, np.nan) - 1) * scale
    elif condition == "ge":
        return (x / y.where(y > 0, np.nan) - 1) * scale
    elif condition == "le":
        return (x / y.where(y < 0, np.nan) - 1) * scale
    else:
        raise ValueError("parameter condition must be 'ne', 'ge', or 'le'.")


def cal_top_corr(x: pd.DataFrame, y: pd.DataFrame, sort_var: pd.DataFrame, top_size: int, ascending: bool = False):
    res = {}
    for code in x.columns:
        df = pd.DataFrame(
            {
                "x": x[code],
                "y": y[code],
                "sv": sort_var[code],
            }
        )
        sorted_data = df.sort_values(by="sv", ascending=ascending)
        top_data = sorted_data.head(top_size)
        res[code] = top_data[["x", "y"]].astype(np.float64).corr(method="spearman").at["x", "y"]
    return pd.Series(res)


# --------------- Algs for factors ---------------


def cal_roll_return(x: pd.Series, ticker_n: str, ticker_d: str, prc_n: str, prc_d: str):
    if x.isnull().any():
        return np.nan
    if x[prc_d] > 0:
        cntrct_d, cntrct_n = x[ticker_d].split("_")[0], x[ticker_n].split("_")[0]
        month_d, month_n = int(cntrct_d[-2:]), int(cntrct_n[-2:])
        dlt_month = (month_d - month_n) % 12
        if dlt_month > 0:
            return np.round((x[prc_n] / x[prc_d] - 1) / dlt_month * 12 * 100, 6)
        else:
            return np.nan
    else:
        return np.nan


def cal_res(y: pd.DataFrame, x: pd.DataFrame) -> pd.Series:
    xyb = (y * x).mean()
    xxb = (x * x).mean()
    xb, yb = x.mean(), y.mean()
    icov = xyb - xb * yb
    ivar = xxb - xb * xb
    beta = robust_div(icov, ivar)
    return y.iloc[-1, :] - beta * x.iloc[-1, :]  # type:ignore


def cal_tres(ret: pd.DataFrame, ts_raw: pd.DataFrame, win: int) -> pd.Series:
    return -cal_res(y=ret.tail(win), x=ts_raw.tail(win))


def cal_ts_by_day(ts_raw: pd.Series, win: int) -> pd.Series:
    return ts_raw.rolling(window=win, min_periods=int(2 * win / 3)).mean()


def cal_ts(ts: pd.DataFrame) -> pd.Series:
    return ts.iloc[-1, :]


def cal_basis(basis_rate: pd.DataFrame, win: int) -> pd.Series:
    return basis_rate.tail(win).mean()


def cal_bres(ret: pd.DataFrame, basis_rate: pd.DataFrame, ws: int, wl: int) -> pd.Series:
    y, x = ret, basis_rate
    res0 = cal_res(y=y.tail(ws), x=x.tail(ws))
    res1 = cal_res(y=y.tail(wl), x=x.tail(wl))
    return res0 - res1


def cal_mtm(ret: pd.DataFrame, to_rate: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    ret_adj = ret * to_rate
    m0 = ret_adj.tail(w0).sum()
    m1 = ret_adj.tail(w1).sum()
    res = m0 * np.sqrt(w1 / w0) - m1
    return res


def cal_minor(ret_minor: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    m0 = ret_minor.tail(w0).mean()
    m1 = ret_minor.tail(w1).mean()
    res = m0 * np.sqrt(w0 / w1) - m1
    return res


def cal_minres(ret: pd.DataFrame, ret_minor: pd.DataFrame, w: int) -> pd.Series:
    m0 = ret.tail(w).mean()
    m1 = ret_minor.tail(w).mean()
    res = m0 - m1
    return res


def cal_rs(stock: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    m0, m1 = stock.tail(w0).mean(), stock.tail(w1).mean()
    s = stock.iloc[-1]
    rs0, rs1 = 1 - s / m0.where(m0 > 0, np.nan), 1 - s / m1.where(m1 > 0, np.nan)
    res = rs0 - rs1
    return res


def cal_val(close: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    m0 = robust_ret(x=close.iloc[-1], y=close.iloc[-w0 - 1], scale=100)
    m1 = robust_ret(x=close.iloc[-1], y=close.iloc[-w1 - 1], scale=100)
    res = m0 * np.sqrt(w1 / w0) - m1
    return res


def cal_liquidity(ret: pd.DataFrame, turnover: pd.DataFrame, w: int) -> pd.Series:
    liq = robust_div(ret * 1e14, turnover)
    res = liq.tail(w).mean()
    return res


def cal_kurt(ret: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    m0 = ret.tail(w0).kurt()
    m1 = ret.tail(w1).kurt()
    res = m1 - m0
    return res


def cal_skew(ret: pd.DataFrame, win: int) -> pd.Series:
    return -ret.tail(win).skew()


def cal_ctp(to_rate: pd.DataFrame, prc: pd.DataFrame, vol: pd.DataFrame, win: int, lbd: float) -> pd.Series:
    top_size = int(win * lbd)
    return -cal_top_corr(x=to_rate.tail(win), y=prc.tail(win), sort_var=vol.tail(win), top_size=top_size)


def cal_cvp(iv: pd.DataFrame, prc: pd.DataFrame, vol: pd.DataFrame, win: int, lbd: float) -> pd.Series:
    top_size = int(win * lbd)
    return -cal_top_corr(x=iv.tail(win), y=prc.tail(win), sort_var=vol.tail(win), top_size=top_size)


def cal_ikurt(ikurt: pd.DataFrame, win: int) -> pd.Series:
    res = -ikurt.tail(win).sum()
    return res


def cal_npc_by_minute(tday_minb_data: pd.DataFrame, ret: str = "ret", vol: str = "vol") -> float:
    pos_data = tday_minb_data.query(f"{ret} > 0")
    neg_data = tday_minb_data.query(f"{ret} < 0")
    lng_vol = pos_data[vol].sum()
    srt_vol = neg_data[vol].sum()
    return lng_vol - srt_vol


def cal_npls(npc: pd.DataFrame, oi: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    aver_oi = oi.rolling(window=2).mean()
    npls = robust_div(npc, aver_oi)
    m0, m1 = npls.tail(w0).sum(), npls.tail(w1).sum()
    return m0 * np.sqrt(w1 / w0) - m1


def cal_reoc_by_minute(tday_minb_data: pd.DataFrame, eff: str = "eff", ret: str = "ret") -> float:
    net_data = tday_minb_data.iloc[1:, :]
    eff_sum = net_data[eff].sum()
    if eff_sum > 0:
        wgt = net_data[eff] / eff_sum
        reoc = net_data[ret].fillna(0) @ wgt * 1e4
        return reoc
    else:
        return 0.0


def cal_reoc(reoc: pd.DataFrame, w0: int, w1: int) -> pd.Series:
    m0, m1 = reoc.tail(w0).sum(), reoc.tail(w1).sum()
    return m0 * np.sqrt(w1 / w0) - m1


def cal_smt_idx(z: pd.Series, ret: str = "ret", vol: str = "vol") -> float:
    if z[vol] > 1:
        return np.abs(z[ret]) / np.log(z[vol]) * 1e4
    else:
        return 0


def preprocess_smt(instru_data_1m: pd.DataFrame) -> pd.DataFrame:
    slc_data = instru_data_1m.dropna(axis=0, how="all")
    slc_data[["turnover", "vol", "ret"]].fillna(0, inplace=True)
    slc_data = slc_data.query("vol >= 0")
    slc_data["smt_idx"] = slc_data[["ret", "vol"]].apply(cal_smt_idx, axis=1)
    slc_data["vp"] = robust_div(slc_data["turnover"], slc_data["vol"])
    slc_data = slc_data.sort_values(by=["trade_day", "smt_idx"], ascending=[True, False])
    return slc_data


def cal_smt_by_minute(sorted_sub_data: pd.DataFrame, lbd: float) -> float:
    tot_data = sorted_sub_data.query("vol > 0")
    tot_amt_sum = tot_data["turnover"].sum()
    if tot_amt_sum > 0:
        tot_w = tot_data["turnover"] / tot_amt_sum
        tot_prc = tot_data["vp"] @ tot_w
    else:
        return np.nan
    volume_threshold = tot_data["vol"].sum() * lbd
    n = sum(tot_data["vol"].cumsum() < volume_threshold) + 1
    smt_data = tot_data.head(n)
    smt_amt_sum = smt_data["turnover"].sum()
    if smt_amt_sum > 0:
        smt_w = smt_data["turnover"] / smt_amt_sum
        smt_prc = smt_data["vp"] @ smt_w
        smt_p = ((smt_prc / tot_prc - 1) * 1e4) if tot_prc > 0 else np.nan
        return smt_p
    else:
        return np.nan


def cal_smt(smt: pd.DataFrame, win: int) -> pd.Series:
    return smt.tail(win).mean()


def cal_clvpa_by_minute(tday_minb_data: pd.DataFrame, vol: str = "vol", ret: str = "ret") -> float:
    size = len(tday_minb_data)
    sv = tday_minb_data[vol].fillna(0).head(size - 1)
    sr = tday_minb_data[ret].abs().shift(-1).fillna(0).head(size - 1)
    if (sv.std() > 0) and (sr.std() > 0):
        return -sv.corr(other=sr)
    else:
        return 0


def cal_clvpa(clvpa: pd.DataFrame, win: int) -> pd.Series:
    return clvpa.tail(win).mean()


def cal_clvpa2_by_minute(tday_minb_data: pd.DataFrame, vol: str = "vol", ret: str = "ret") -> float:
    size = len(tday_minb_data)
    sv = tday_minb_data[vol].fillna(0).head(size - 1)
    sr = tday_minb_data[ret].shift(-1).fillna(0).head(size - 1)
    if (sv.std() > 0) and (sr.std() > 0):
        return -sv.corr(other=sr)
    else:
        return 0


def cal_clvpa2(clvpa2: pd.DataFrame, win: int) -> pd.Series:
    return clvpa2.tail(win).mean()


def cal_clvpr_by_minute(tday_minb_data: pd.DataFrame, vol: str = "vol", ret: str = "ret") -> float:
    size = len(tday_minb_data)
    sv = tday_minb_data[vol].shift(-1).fillna(0).head(size - 1)
    sr = tday_minb_data[ret].abs().fillna(0).head(size - 1)
    if (sv.std() > 0) and (sr.std() > 0):
        return -sv.corr(other=sr)
    else:
        return 0


def cal_clvpr(clvpr: pd.DataFrame, win: int) -> pd.Series:
    return clvpr.tail(win).mean()
