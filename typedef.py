from itertools import product
from dataclasses import dataclass
from typing import Literal
from typedef_factor import CCfgFactors


@dataclass(frozen=True)
class CCfgInstru:
    sectorL0: Literal["C", "E"]  # C = commodity, E = Equity
    sectorL1: Literal["AUG", "MTL", "BLK", "OIL", "CHM", "AGR"]


TInstruName = str
TUniverse = dict[TInstruName, CCfgInstru]
TSectors = list[str]


@dataclass(frozen=True)
class CCfgAvlb:
    window: int
    threshold: float
    keep: int
    oi_cap_ratio: float

    @property
    def lag(self) -> int:
        return max(self.window, self.keep) * 2


@dataclass(frozen=True)
class CCfgCss:
    vma_win: int
    vma_threshold: float
    vma_wgt: float


@dataclass(frozen=True)
class CCfgICov:
    win: int

@dataclass(frozen=True)
class CCfgQSim:
    win: int

@dataclass(frozen=True)
class CCfgOptimizer:
    window: int
    lbd: float

@dataclass(frozen=True)
class CCfgProj:
    pid: str
    vid: str
    project_data_dir: str
    path_calendar: str
    codes: list[str]
    avlb: CCfgAvlb
    css: CCfgCss
    icov: CCfgICov
    factors: CCfgFactors
    qsim: CCfgQSim
    optimizer: CCfgOptimizer
    tgt_rets: list[str]
    cost_rate_sub: float
    cost_rate_pri: float
    init_cash: float

    @property
    def sim_codes_fac(self) -> list[str]:
        return [f"{fac}-{ret}" for fac, ret in product(self.factors.to_list(), self.tgt_rets)]

    @property
    def sim_codes_stg(self) -> list[str]:
        return [f"{ret}-{ret}" for ret in self.tgt_rets]

    @property
    def secondary_codes(self) -> list[str]:
        return [z.replace("9999", "8888") for z in self.codes]


@dataclass(frozen=True)
class CCfgTables:
    avlb: str
    css: str
    icov: str
    srets: str
    fac_raw: str
    fac_nrm: str
    sig_fac: str
    sig_stg: str
    sim_fac: str
    optimize: str


@dataclass(frozen=True)
class CCfgDbs:
    public: str
    basic: str
    user: str


@dataclass(frozen=True)
class CSimArgs:
    sig: str
    ret: str

    @property
    def save_id(self) -> str:
        return f"{self.sig}-{self.ret}"
