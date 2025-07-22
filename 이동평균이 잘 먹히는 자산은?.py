# ───────────────────────────────────────────────────────────
# 0. 필수 패키지 설치 (코랩에서 한 번만 실행)
# ───────────────────────────────────────────────────────────
!pip install yfinance pandas numpy --quiet

# ───────────────────────────────────────────────────────────
# 1. 라이브러리 임포트 및 경고 억제
# ───────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.display import display

# ───────────────────────────────────────────────────────────
# 2. 설정값
# ───────────────────────────────────────────────────────────
TREND_FOLLOWING_TICKERS_UPDATED = [
    # 기존 목록 중 제외되지 않은 주식
    "NVDA", "TSLA", "AAPL", "META", "NFLX", "GE", "AMZN", "BAC", "CAT",
    "MSFT", "GOOGL", "AMD", "PLTR", "SMCI", "AVGO",

    # 기존 목록 중 제외되지 않은 암호화폐
    "BTC-USD", "ETH-USD", "SOL-USD", "TON-USD", "XRP-USD", "LINK-USD",

    # 새롭게 추가된 주식 (50개 이상)
    "ASML", "CRM", "INTU", "ORCL", "SNPS", "V", "MA", "ADBE", "ACN", "NOW", # 소프트웨어/클라우드/핀테크
    "CDNS", "PANW", "CRWD", "OKTA", "FTNT", "DDOG", "NET", "ZS", "MRVL", "KLAC", # 사이버 보안/반도체 장비
    "HD", "LOW", "DE", "CAT", "UNP", "CP", "NSC", "CSX", "KSU", "UPS", # 산업/운송
    "NEE", "DUK", "SO", "AEP", "SRE", "PCG", "XOM", "CVX", "COP", "EOG", # 유틸리티/에너지
    "JPM", "GS", "MS", "C", "WFC", "BLK", "SPG", "PLD", "EQIX", "AMT", # 금융/부동산
    "ISRG", "TMO", "ABBV", "UNH", "ELV", "CVS", "PFE", "MRK", "BMY", "GSK", # 헬스케어/제약
    "PEP", "KO", "PG", "JNJ", "COST", "WMT", "HD", "MCD", "SBUX", "NKE", # 필수 소비재/소매
    "LMT", "RTX", "NOC", "GD", "BA", # 방위 산업

    # 새롭게 추가된 암호화폐 (20개 이상)
    "BNB-USD", "ADA-USD", "DOT-USD", "AVAX-USD", "UNI-USD", "LTC-USD", "BCH-USD", "XLM-USD", # 주요 알트코인
    "TRX-USD", "DAI-USD", "MKR-USD", "AAVE-USD", "COMP-USD", "SNX-USD", "CRV-USD", # DeFi 관련
    "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", # 메타버스/NFT 관련
    "GRT-USD", "CHZ-USD", "FLOW-USD", "XTZ-USD", "VET-USD" # 기타 유망 코인
]


START_DATE = "2000-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
MA_GRID    = range(10, 260, 10)
MA_TYPES   = ["SMA","EMA","HMA","WMA","VWMA","TEMA"]
COMM_RATE  = 0.0005    # 커미션 0.05%
SLIP_RATE  = 0.0002    # 슬리피지 0.02%

# ───────────────────────────────────────────────────────────
# 3. 보조 함수: 다양한 MA
# ───────────────────────────────────────────────────────────
def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period+1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def hma(series: pd.Series, period: int) -> pd.Series:
    half = period//2
    sq   = int(np.sqrt(period))
    return wma(2*wma(series, half) - wma(series, period), sq)

def vwma(series: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    num = (series * volume).rolling(period).sum()
    den = volume.rolling(period).sum()
    return num/den

def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return 3*e1 - 3*e2 + e3

def get_ma(series: pd.Series, volume: pd.Series, period: int, ma_type: str) -> pd.Series:
    mt = ma_type.upper()
    if mt == "SMA":   return series.rolling(period).mean()
    if mt == "EMA":   return series.ewm(span=period, adjust=False).mean()
    if mt == "HMA":   return hma(series, period)
    if mt == "WMA":   return wma(series, period)
    if mt == "VWMA":  return vwma(series, volume, period)
    if mt == "TEMA":  return tema(series, period)
    raise ValueError(f"Unsupported MA type: {ma_type}")

# ───────────────────────────────────────────────────────────
# 4. Sortino 함수
# ───────────────────────────────────────────────────────────
def sortino(returns: pd.Series, target: float = 0.0) -> float:
    r = returns.dropna().values
    if r.size == 0: return np.nan
    mu       = np.mean(r - target)
    downside = np.minimum(0, r - target)
    down_var = np.mean(downside**2)
    if down_var == 0: return np.nan
    return mu / np.sqrt(down_var) * np.sqrt(252)

# ───────────────────────────────────────────────────────────
# 5. 백테스트 + 비용 + 안정성 평가 + 현재 포지션 + 이격도
# ───────────────────────────────────────────────────────────
records = []

for ticker in TICKERS:
    df     = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)[["Close","Volume"]].dropna()
    close  = df["Close"]
    volume = df["Volume"].ffill()
    ret    = close.pct_change()

    best_adj_sr, best_ma, best_type = -np.inf, None, None

    # ── 그리드 서치: Adjusted_Sortino 최대화 ──
    for ma_type in MA_TYPES:
        for n in MA_GRID:
            ma           = get_ma(close, volume, n, ma_type).reindex(close.index)
            pos          = (close >= ma).astype(int)
            diff         = pos.diff().fillna(0)
            cost         = diff.abs() * (COMM_RATE + SLIP_RATE)
            strat        = pos.shift(1) * ret - cost
            strat_clean  = strat.dropna()
            length       = len(strat_clean)
            segs         = [
                strat_clean.iloc[:length//3],
                strat_clean.iloc[length//3:2*length//3],
                strat_clean.iloc[2*length//3:]
            ]
            sr_list      = [sortino(seg) for seg in segs]
            sr_std       = np.nanstd(sr_list, ddof=0)
            years        = (close.index.max() - close.index.min()).days / 365
            base_sr      = sortino(strat_clean)
            adj_sr       = base_sr * np.sqrt(years) - 1.5 * sr_std

            if np.isfinite(adj_sr) and adj_sr > best_adj_sr:
                best_adj_sr, best_ma, best_type = adj_sr, n, ma_type

    # ── 최적 MA와 포지션 계산 ──
    ma_opt    = get_ma(close, volume, best_ma, best_type).reindex(close.index).ffill()
    pos_opt   = (close >= ma_opt).astype(int)

    # ── 현재 포지션 ──
    price_last = close.iloc[-1]
    ma_last    = ma_opt.iloc[-1]
    # 값이 Series일 때 .item() 처리
    if isinstance(price_last, pd.Series): price_last = price_last.item()
    if isinstance(ma_last,    pd.Series): ma_last    = ma_last.item()
    current_position = "Buy" if price_last >= ma_last else "Cash"

    # ── 이격도(%) 계산 ──
    deviation_pct = (price_last / ma_last - 1) * 100

    # ── 전체 전략 성과 계산 ──
    strat_all = pos_opt.shift(1)*ret - (pos_opt.diff().fillna(0).abs()*(COMM_RATE+SLIP_RATE))
    strat_all = strat_all.dropna()
    n_all     = len(strat_all)
    segs_all  = [
        strat_all.iloc[: n_all//3],
        strat_all.iloc[n_all//3:2*n_all//3],
        strat_all.iloc[2*n_all//3:]
    ]
    sr_all    = [sortino(seg) for seg in segs_all]
    sr_std_all= np.nanstd(sr_all, ddof=0)
    years_all = (close.index.max() - close.index.min()).days / 365
    final_sr  = sortino(strat_all)
    final_adj = final_sr * np.sqrt(years_all) - 2 * sr_std_all

    records.append({
        "Ticker":            ticker,
        "Years":             years_all,
        "Best_MA_Type":      best_type,
        "Best_Period":       best_ma,
        "Sortino":           final_sr,
        "Adjusted_Sortino":  final_adj,
        "SR_Seg1":           sr_all[0],
        "SR_Seg2":           sr_all[1],
        "SR_Seg3":           sr_all[2],
        "SR_Std":            sr_std_all,
        "Current_Position":  current_position,
        "Deviation(%)":      deviation_pct
    })

# ───────────────────────────────────────────────────────────
# 6. 결과 정리 및 출력
# ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)

df = df[df['SR_Std'] <= 4].dropna()

df = df.sort_values("Adjusted_Sortino", ascending=False).reset_index(drop=True)

fmt = {
    "Years":            "{:.2f}",
    "Sortino":          "{:.3f}",
    "SR_Seg1":          "{:.3f}",
    "SR_Seg2":          "{:.3f}",
    "SR_Seg3":          "{:.3f}",
    "SR_Std":           "{:.3f}",
    "Adjusted_Sortino": "{:.3f}",
    "Deviation(%)":     "{:.2f}"
}

display(
    df.style
      .format(fmt)
      .set_caption("추세추종 전략 성과 및 현재 포지션 (Adjusted Sortino 최대화)")
      .set_table_styles([
          {'selector':'th','props':[('text-align','center')]},
          {'selector':'td','props':[('text-align','center')]}
      ])
)
