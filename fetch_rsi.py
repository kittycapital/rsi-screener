"""
S&P 500 RSI Screener
Finds oversold (RSI < 20) and overbought (RSI > 80) signals
and calculates 63-day forward returns.
Generates per-ticker chart JSON for Lightweight Charts.
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = 'data.json'
CHARTS_DIR = 'charts'
LOOKBACK_YEARS = 3
RSI_PERIOD = 14
FORWARD_DAYS = 63  # ~3 months
OVERSOLD_THRESHOLD = 20
OVERBOUGHT_THRESHOLD = 80


def get_sp500_tickers():
    """Return S&P 500 tickers + popular ETFs"""
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
    sp500 = [
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
        'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP',
        'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV',
        'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA',
        'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BIO',
        'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX',
        'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE',
        'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD',
        'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI',
        'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB',
        'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT',
        'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD',
        'DE', 'DECK', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
        'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN',
        'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS',
        'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG',
        'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE',
        'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG',
        'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA',
        'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL',
        'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX',
        'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM',
        'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ',
        'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC',
        'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH',
        'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV',
        'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP',
        'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX',
        'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK',
        'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI',
        'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW',
        'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG',
        'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW',
        'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PYPL',
        'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK',
        'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SE',
        'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI',
        'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF',
        'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC',
        'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV',
        'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER',
        'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI',
        'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ',
        'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB',
        'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM',
        'ZBH', 'ZBRA', 'ZTS'
    ]
    tickers = list(dict.fromkeys(etfs + sp500))
    print(f"  Loaded {len(tickers)} tickers (S&P 500 + ETFs)")
    return tickers


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fetch_stock_data(tickers, years=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + FORWARD_DAYS + 30)
    print(f"  Fetching data from {start_date.date()} to {end_date.date()}")

    all_data = {}
    all_ohlc = {}
    failed = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = ' '.join(batch)
        try:
            data = yf.download(batch_str, start=start_date, end=end_date,
                             progress=False, threads=True)
            if len(batch) == 1:
                if not data.empty:
                    all_data[batch[0]] = data['Close']
                    all_ohlc[batch[0]] = data[['Open', 'High', 'Low', 'Close']].copy()
            else:
                for ticker in batch:
                    try:
                        if ticker in data['Close'].columns:
                            ticker_close = data['Close'][ticker].dropna()
                            if len(ticker_close) > RSI_PERIOD + FORWARD_DAYS:
                                all_data[ticker] = ticker_close
                                ohlc_df = pd.DataFrame({
                                    'Open': data['Open'][ticker],
                                    'High': data['High'][ticker],
                                    'Low': data['Low'][ticker],
                                    'Close': data['Close'][ticker]
                                }).dropna()
                                all_ohlc[ticker] = ohlc_df
                    except:
                        failed.append(ticker)
            print(f"  Processed {min(i + batch_size, len(tickers))}/{len(tickers)} tickers")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Batch error: {e}")
            failed.extend(batch)

    print(f"  Successfully fetched {len(all_data)} tickers, {len(failed)} failed")
    return all_data, all_ohlc


def find_signals(all_data):
    oversold_signals = []
    overbought_signals = []
    current_oversold = []
    current_overbought = []
    ticker_signals = {}

    today = datetime.now().date()
    three_years_ago = today - timedelta(days=LOOKBACK_YEARS * 365)

    for ticker, prices in all_data.items():
        try:
            rsi = calculate_rsi(prices, RSI_PERIOD)
            df = pd.DataFrame({'close': prices, 'rsi': rsi}).dropna()
            if len(df) < FORWARD_DAYS + 10:
                continue

            df['forward_return'] = df['close'].shift(-FORWARD_DAYS) / df['close'] - 1
            df = df[df.index >= pd.Timestamp(three_years_ago)]

            ticker_signals[ticker] = []

            # Oversold
            oversold = df[df['rsi'] < OVERSOLD_THRESHOLD]
            for idx, row in oversold.iterrows():
                signal = {
                    'ticker': ticker,
                    'date': idx.strftime('%Y-%m-%d'),
                    'price': round(row['close'], 2),
                    'rsi': round(row['rsi'], 1),
                    'type': 'oversold'
                }
                ticker_signals[ticker].append({
                    'date': signal['date'], 'type': 'oversold', 'rsi': signal['rsi']
                })
                if pd.notna(row['forward_return']):
                    signal['forward_return'] = round(row['forward_return'] * 100, 2)
                    oversold_signals.append(signal)

            # Overbought
            overbought = df[df['rsi'] > OVERBOUGHT_THRESHOLD]
            for idx, row in overbought.iterrows():
                signal = {
                    'ticker': ticker,
                    'date': idx.strftime('%Y-%m-%d'),
                    'price': round(row['close'], 2),
                    'rsi': round(row['rsi'], 1),
                    'type': 'overbought'
                }
                ticker_signals[ticker].append({
                    'date': signal['date'], 'type': 'overbought', 'rsi': signal['rsi']
                })
                if pd.notna(row['forward_return']):
                    signal['forward_return'] = round(row['forward_return'] * 100, 2)
                    overbought_signals.append(signal)

            # Current
            latest = df.iloc[-1]
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            latest_rsi = latest['rsi']
            latest_price = latest['close']

            if latest_rsi < OVERSOLD_THRESHOLD:
                current_oversold.append({
                    'ticker': ticker, 'date': latest_date,
                    'price': round(latest_price, 2), 'rsi': round(latest_rsi, 1)
                })
            elif latest_rsi > OVERBOUGHT_THRESHOLD:
                current_overbought.append({
                    'ticker': ticker, 'date': latest_date,
                    'price': round(latest_price, 2), 'rsi': round(latest_rsi, 1)
                })

        except Exception:
            continue

    return oversold_signals, overbought_signals, current_oversold, current_overbought, ticker_signals


def get_top_signals(signals, top_n=10, best=True):
    if not signals:
        return []
    return sorted(signals, key=lambda x: x.get('forward_return', 0), reverse=best)[:top_n]


def calculate_statistics(signals):
    if not signals:
        return {'count': 0, 'avg_return': 0, 'win_rate': 0}
    returns = [s['forward_return'] for s in signals if 'forward_return' in s]
    if not returns:
        return {'count': 0, 'avg_return': 0, 'win_rate': 0}
    wins = len([r for r in returns if r > 0])
    return {
        'count': len(returns),
        'avg_return': round(np.mean(returns), 2),
        'median_return': round(np.median(returns), 2),
        'win_rate': round(wins / len(returns) * 100, 1),
        'best_return': round(max(returns), 2),
        'worst_return': round(min(returns), 2)
    }


def save_chart_files(all_ohlc, all_data, ticker_signals):
    """Save individual chart JSON files for Lightweight Charts"""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    saved_count = 0

    for ticker in all_ohlc.keys():
        try:
            ohlc_df = all_ohlc[ticker].copy()
            prices = all_data.get(ticker)
            if prices is None or len(ohlc_df) < 50:
                continue

            rsi = calculate_rsi(prices, RSI_PERIOD)

            # Keep last ~1.5 years for chart display
            ohlc_df = ohlc_df.tail(400)

            ohlc_data = []
            for idx, row in ohlc_df.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                rsi_value = rsi.get(idx, None)
                ohlc_data.append({
                    'time': date_str,
                    'open': round(row['Open'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'close': round(row['Close'], 2),
                    'rsi': round(rsi_value, 1) if pd.notna(rsi_value) else None
                })

            # Only include signals within the chart date range
            chart_start = ohlc_data[0]['time']
            signals = [s for s in ticker_signals.get(ticker, []) if s['date'] >= chart_start]

            chart_data = {
                'ticker': ticker,
                'ohlc': ohlc_data,
                'signals': signals
            }

            filepath = os.path.join(CHARTS_DIR, f'{ticker}.json')
            with open(filepath, 'w') as f:
                json.dump(chart_data, f, separators=(',', ':'))  # compact JSON

            saved_count += 1
        except Exception:
            continue

    print(f"  Saved {saved_count} chart files")
    return saved_count


def main():
    print("=" * 50)
    print("S&P 500 RSI Screener")
    print("=" * 50)

    print("\n1. Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()

    print("\n2. Downloading price data...")
    all_data, all_ohlc = fetch_stock_data(tickers, LOOKBACK_YEARS)

    print("\n3. Analyzing RSI signals...")
    oversold_signals, overbought_signals, current_oversold, current_overbought, ticker_signals = find_signals(all_data)

    print(f"  Found {len(oversold_signals)} oversold signals (RSI < {OVERSOLD_THRESHOLD})")
    print(f"  Found {len(overbought_signals)} overbought signals (RSI > {OVERBOUGHT_THRESHOLD})")
    print(f"  Currently oversold: {len(current_oversold)} stocks")
    print(f"  Currently overbought: {len(current_overbought)} stocks")

    print("\n4. Selecting top signals...")
    top_oversold = get_top_signals(oversold_signals, 10, best=True)
    top_overbought = get_top_signals(overbought_signals, 10, best=False)

    oversold_stats = calculate_statistics(oversold_signals)
    overbought_stats = calculate_statistics(overbought_signals)

    print("\n5. Saving chart files...")
    save_chart_files(all_ohlc, all_data, ticker_signals)

    available_tickers = sorted([t for t in all_ohlc.keys() if t in all_data])

    output = {
        'topOversold': top_oversold,
        'topOverbought': top_overbought,
        'currentOversold': sorted(current_oversold, key=lambda x: x['rsi'])[:20],
        'currentOverbought': sorted(current_overbought, key=lambda x: x['rsi'], reverse=True)[:20],
        'oversoldStats': oversold_stats,
        'overboughtStats': overbought_stats,
        'availableTickers': available_tickers,
        'config': {
            'rsiPeriod': RSI_PERIOD,
            'forwardDays': FORWARD_DAYS,
            'oversoldThreshold': OVERSOLD_THRESHOLD,
            'overboughtThreshold': OVERBOUGHT_THRESHOLD,
            'lookbackYears': LOOKBACK_YEARS
        },
        'lastUpdated': datetime.utcnow().isoformat() + 'Z'
    }

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"\n📉 Oversold: {oversold_stats['count']} signals | Avg: {oversold_stats['avg_return']}% | Win: {oversold_stats['win_rate']}%")
    print(f"📈 Overbought: {overbought_stats['count']} signals | Avg: {overbought_stats['avg_return']}% | Win: {overbought_stats['win_rate']}%")

    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to {DATA_FILE}")


if __name__ == '__main__':
    main()
