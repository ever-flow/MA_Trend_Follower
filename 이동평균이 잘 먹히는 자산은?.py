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
TICKERS = [
    "AAPL", "NFLX", "NVDA", "BTC-USD", "FTM-USD", "TSLA", "XRP-USD", "CAT",
    "BNB-USD", "ISRG", "AMD", "GOOGL", "XLM-USD", "TRX-USD", "^BVSP",
    "ADA-USD", "VUG", "AMZN", "ENJ-USD", "META", "QQQ", "MANA-USD",
    "ETH-USD", "VET-USD", "GRT-USD", "SPYG", "GS", "VOO", "BA", "IVV",
    "BLK", "CRM", "AVGO", "SMCI", "MKR-USD", "ASML", "CDNS", "MRVL",
    "^GDAXI", "EQIX", "QUAL", "OJ=F", "ZS", "MA", "GD", "GC=F", "NOC",
    "AMT", "SPG", "V", "TMO", "AVAX-USD", "LMT", "SOL-USD", "SBUX",
    "LINK-USD", "PA=F", "AEP", "MTUM", "PANW", "CP", "RNDR-USD", "ETN",
    "JASMY-USD", "PGR", "068270.KS", "ORLY", "TQQQ", "XLK", "051910.KS",
    "035720.KS", "005930.KS", "GC=F", "ROP", "035420.KS", "XLI", "SOXL",
    "SI=F"
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
# 5. 메인 루프 (에러 스킵)
# ───────────────────────────────────────────────────────────
records = []
errors  = []

for ticker in TICKERS:
    try:
        # 5-1. 데이터 다운로드
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)[["Close","Volume"]].dropna()
        if df.empty:
            raise ValueError("Empty dataframe returned")

        close  = df["Close"]
        volume = df["Volume"].ffill()
        ret    = close.pct_change()

        best_adj_sr, best_ma, best_type = -np.inf, None, None

        # 5-2. 그리드 서치
        for ma_type in MA_TYPES:
            for n in MA_GRID:
                ma           = get_ma(close, volume, n, ma_type).reindex(close.index)
                pos          = (close >= ma).astype(int)
                diff         = pos.diff().fillna(0)
                cost         = diff.abs() * (COMM_RATE + SLIP_RATE)
                strat        = pos.shift(1) * ret - cost
                strat_clean  = strat.dropna()
                length       = len(strat_clean)
                if length < 30:   # 너무 짧으면 패스
                    continue

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

        if best_ma is None or best_type is None:
            raise RuntimeError("No valid MA combination found")

        # 5-3. 최적 MA로 최종 성과 계산
        ma_opt    = get_ma(close, volume, best_ma, best_type).reindex(close.index).ffill()
        pos_opt   = (close >= ma_opt).astype(int)

        price_last = float(close.iloc[-1])
        ma_last    = float(ma_opt.iloc[-1])

        current_position = "Buy" if price_last >= ma_last else "Cash"
        deviation_pct    = (price_last / ma_last - 1) * 100

        strat_all = pos_opt.shift(1)*ret - (pos_opt.diff().fillna(0).abs()*(COMM_RATE+SLIP_RATE))
        strat_all = strat_all.dropna()
        n_all     = len(strat_all)
        if n_all < 30:
            raise RuntimeError("Too few data points after cleaning")

        segs_all  = [
            strat_all.iloc[: n_all//3],
            strat_all.iloc[n_all//3:2*n_all//3],
            strat_all.iloc[2*n_all//3:]
        ]
        sr_all       = [sortino(seg) for seg in segs_all]
        sr_std_all   = np.nanstd(sr_all, ddof=0)
        years_all    = (close.index.max() - close.index.min()).days / 365
        final_sr     = sortino(strat_all)
        final_adj    = final_sr * np.sqrt(years_all) - 2 * sr_std_all

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

    except Exception as e:
        errors.append((ticker, str(e)))
        continue

# ───────────────────────────────────────────────────────────
# 6. 결과 정리 및 출력
# ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)

# 너무 불안정한 전략 제거 & NaN 제거
df = df[df['SR_Std'] <= 4].dropna(subset=["Adjusted_Sortino"])

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

# ───────────────────────────────────────────────────────────
# 7. 에러 티커 요약 출력
# ───────────────────────────────────────────────────────────
if errors:
    print("\n[SKIPPED TICKERS & REASONS]")
    for t, msg in errors:
        print(f"- {t}: {msg}")
else:
    print("\nNo errors 🎉")
