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
TICKERS_RAW = [ 
    'AAPL', 'AXON', 'NFLX', 'NVDA', 'DECK', '066970.KS', 'BTC-USD', 'TSLA',
    'CAT', 'FET-USD', '^BVSP', 'AMD', 'GOOGL', 'HEI', 'ETN', 'VUG',
    'DXCM', 'AMZN', 'ISRG', '005430.KS', 'META', 'FICO', 'QQQ', 'ORLY',
    'MELI', 'DOGE-USD', 'WING', 'FTM-USD', 'KKR', 'GS', 'CFX-USD', 'XLK',
    'TQQQ', 'MANA-USD', 'NVO', '009450.KS', 'PGR', '035900.KS', 'BLK',
    'XRP-USD', 'ON', 'SPYG', 'VOO', 'CRM', '051910.KS', 'BNB-USD', 'BYON',
    'AVGO', 'BA', 'IVV', 'MRVL', 'ETH-USD','MSTR', 'BOTZ', 'BLOK'
]
# 중복 제거 while preserving order
seen = set()
TICKERS = []
for t in TICKERS_RAW:
    if t not in seen:
        seen.add(t)
        TICKERS.append(t)

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
MA_GRID = range(10, 260, 10)
MA_TYPES = ["SMA", "EMA", "HMA", "WMA", "VWMA", "TEMA"]
COMM_RATE = 0.0005  # 커미션 0.05%
SLIP_RATE = 0.0002  # 슬리피지 0.02%

# ───────────────────────────────────────────────────────────
# 3. 보조 함수: 다양한 MA
# ───────────────────────────────────────────────────────────
def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, period: int) -> pd.Series:
    half = period // 2
    sq = int(np.sqrt(period))
    return wma(2 * wma(series, half) - wma(series, period), sq)

def vwma(series: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    num = (series * volume).rolling(period).sum()
    den = volume.rolling(period).sum()
    return num / den

def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3

def get_ma(series: pd.Series, volume: pd.Series, period: int, ma_type: str) -> pd.Series:
    mt = ma_type.upper()
    if mt == "SMA": return series.rolling(period).mean()
    if mt == "EMA": return series.ewm(span=period, adjust=False).mean()
    if mt == "HMA": return hma(series, period)
    if mt == "WMA": return wma(series, period)
    if mt == "VWMA": return vwma(series, volume, period)
    if mt == "TEMA": return tema(series, period)
    raise ValueError(f"Unsupported MA type: {ma_type}")

# ───────────────────────────────────────────────────────────
# 4. Sortino 함수
# ───────────────────────────────────────────────────────────
def sortino(returns: pd.Series, target: float = 0.0) -> float:
    r = returns.dropna().values
    if r.size == 0:
        return np.nan
    mu = np.mean(r - target)
    downside = np.minimum(0, r - target)
    down_var = np.mean(downside ** 2)
    if down_var == 0:
        return np.nan
    return mu / np.sqrt(down_var) * np.sqrt(252)

# ───────────────────────────────────────────────────────────
# 5. 메인 루프 (에러 스킵)
# ───────────────────────────────────────────────────────────
records = []
errors = []

for ticker in TICKERS:
    try:
        # 5-1. 데이터 다운로드 (auto_adjust=True)
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)[["Close", "Volume"]].dropna()
        if df.empty:
            raise ValueError("Empty dataframe returned")
        close = df["Close"]
        volume = df["Volume"].ffill()
        ret = close.pct_change()
        # 전체 기간 길이 체크: 최소 4.5년 이상
        total_years = (close.index.max() - close.index.min()).days / 365
        if total_years < 4.5:
            raise RuntimeError(f"Data span too short: {total_years:.2f} years (<4.5)")

        best_adj_sr, best_ma, best_type = -np.inf, None, None

        # 5-2. 그리드 서치
        for ma_type in MA_TYPES:
            for n in MA_GRID:
                ma = get_ma(close, volume, n, ma_type).reindex(close.index)
                pos = (close >= ma).astype(int)
                diff = pos.diff().fillna(0)
                cost = diff.abs() * (COMM_RATE + SLIP_RATE)
                strat = pos.shift(1) * ret - cost
                strat_clean = strat.dropna()
                length = len(strat_clean)
                if length < 100:
                    continue  # 최소 데이터 포인트

                # 겹치는 5개의 rolling window: each window length = floor(length/3)
                window = length // 3
                if window < int(1.5 * 252):  # 대략 1.5년 거래일 기준 (252일/년)
                    continue  # 이 조합의 window가 너무 짧으면 skip

                step = (length - window) // 4
                if step <= 0:
                    continue  # 겹치게 만들 수 없음

                sr_list = []
                for i in range(5):
                    start = i * step
                    end = start + window
                    if i == 4:  # 마지막은 반드시 끝까지 맞추기
                        end = length
                        start = end - window
                    segment = strat_clean.iloc[start:end]
                    sr_list.append(sortino(segment))

                if any((np.isnan(sr) or sr <= 0) for sr in sr_list):
                    continue  # 모든 window에서 양수 필요

                sr_std = np.nanstd(sr_list, ddof=0)
                base_sr = sortino(strat_clean)
                adj_sr = base_sr * np.sqrt(total_years) - 3 * sr_std  # 페널티 계수 3

                if np.isfinite(adj_sr) and adj_sr > best_adj_sr:
                    best_adj_sr, best_ma, best_type = adj_sr, n, ma_type

        if best_ma is None:
            raise RuntimeError("No valid MA combination found")

        # 5-3. 최적 MA로 최종 성과 계산
        ma_opt = get_ma(close, volume, best_ma, best_type).reindex(close.index).ffill()
        pos_opt = (close >= ma_opt).astype(int)
        price_last = float(close.iloc[-1])
        ma_last = float(ma_opt.iloc[-1])
        current_position = "Buy" if price_last >= ma_last else "Cash"
        deviation_pct = (price_last / ma_last - 1) * 100
        strat_all = pos_opt.shift(1) * ret - (pos_opt.diff().fillna(0).abs() * (COMM_RATE + SLIP_RATE))
        strat_all = strat_all.dropna()
        n_all = len(strat_all)
        if n_all < 100:
            raise RuntimeError("Too few data points after cleaning")

        # ── ⓐ Sortino 관련 지표: 겹치는 5개 rolling window
        window_all = n_all // 3
        if window_all < int(1.5 * 252):
            raise RuntimeError("Overall windows too short for stable segmentation")

        step_all = (n_all - window_all) // 4
        if step_all <= 0:
            raise RuntimeError("Cannot construct overlapping windows for final strat")

        sr_all = []
        for i in range(5):
            start = i * step_all
            end = start + window_all
            if i == 4:
                end = n_all
                start = end - window_all
            segment = strat_all.iloc[start:end]
            sr_all.append(sortino(segment))
        sr_std_all = float(np.nanstd(sr_all, ddof=0))
        years_all = (close.index.max() - close.index.min()).days / 365
        final_sr = float(sortino(strat_all))
        final_adj = float(final_sr * np.sqrt(years_all) - 3 * sr_std_all)

        # ── ⓑ CAGR & MDD
        equity = (strat_all + 1).cumprod()
        cagr = float(equity.iloc[-1] ** (1 / years_all) - 1)
        # Max Drawdown: equity / cummax -1 gives negatives, take min
        max_dd_pct = float((equity / equity.cummax() - 1).min() * 100)  # 음수
        # Calmar Ratio = CAGR / |MaxDD|
        calmar_ratio = cagr / (abs(max_dd_pct) / 100) if max_dd_pct != 0 else np.nan

        # 리스트 요소 정리
        SRs = [float(x) if x is not None else np.nan for x in sr_all]
        records.append({
            "Ticker": ticker,
            "Years": float(years_all),
            "MA_Type": best_type,
            "Period": int(best_ma),
            "Sortino": final_sr,
            "AdjSortino": final_adj,
            "SR1": SRs[0],
            "SR2": SRs[1],
            "SR3": SRs[2],
            "SR4": SRs[3],
            "SR5": SRs[4],
            "SR_Std": sr_std_all,
            "CAGR": cagr * 100,
            "MaxDD": max_dd_pct,  # 이미 % 형태, 음수
            "Calmar": calmar_ratio,
            "Position": current_position,
            "Deviation": deviation_pct
        })

    except Exception as e:
        errors.append((ticker, str(e)))
        continue

# ───────────────────────────────────────────────────────────
# 6. 결과 정리 및 출력
# ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)
# 너무 불안정한 전략 제거 & NaN 제거
df = df[df['SR_Std'] <= 4].dropna()
# 숫자 컬럼 강제 float 캐스팅
num_cols = ["Years", "Sortino", "SR1", "SR2", "SR3", "SR4", "SR5", "SR_Std",
            "AdjSortino", "Deviation", "CAGR", "MaxDD", "Calmar"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
df = df.sort_values("AdjSortino", ascending=False).reset_index(drop=True)

# 포맷 간소화
fmt = {
    "Years": "{:.2f}",
    "Sortino": "{:.3f}",
    "AdjSortino": "{:.3f}",
    "SR1": "{:.3f}",
    "SR2": "{:.3f}",
    "SR3": "{:.3f}",
    "SR4": "{:.3f}",
    "SR5": "{:.3f}",
    "SR_Std": "{:.3f}",
    "Deviation": "{:.2f}",
    "CAGR": "{:.2f}",
    "MaxDD": "{:.2f}",
    "Calmar": "{:.2f}"
}
fmt_existing = {k: v for k, v in fmt.items() if k in df.columns}
display(
    df.style
    .format(fmt_existing)
    .set_caption("추세추종 전략 성과 및 현재 포지션 (Adjusted Sortino 최대화)") 
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
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
