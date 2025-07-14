import pandas as pd
import numpy as np
import talib
from binance.client import Client
from datetime import datetime, timedelta
import time
import os

# --- AYARLAR VE SABİTLER ---
API_KEY = ""
API_SECRET = ""
COIN_LIST = [
    'BTCUSDT', 'ASRUSDT', 'ETHUSDT'
]

HISTORY_BUFFER = 50
COMMISSION_RATE = 0.00045

client = Client(API_KEY, API_SECRET)
print("Binance istemcisi başarıyla başlatıldı.")

# --- YARDIMCI FONKSİYONLAR ---
def get_binance_data(symbol, interval, start_date_str, limit=1000):
    try:
        klines = client.futures_historical_klines(symbol, interval, start_date_str, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
        df = df[['OpenTime', 'Open', 'High', 'Low', 'Close']]
        df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close']: df[col] = pd.to_numeric(df[col])
        return df
    except Exception as e:
        print(f"Veri çekme hatası ({symbol}, {interval}): {e}"); return None

def get_funding_fee_percent(symbol, entry_time, exit_time, position_type):
    try:
        start_ms, end_ms = int(entry_time.timestamp() * 1000), int(exit_time.timestamp() * 1000)
        funding_history = client.futures_funding_rate(symbol=symbol, startTime=start_ms, endTime=end_ms, limit=1000)
        if not funding_history: return 0.0
        total_funding_rate = sum(float(rate['fundingRate']) for rate in funding_history)
        return -total_funding_rate if position_type == 'LONG' else total_funding_rate
    except Exception as e:
        print(f"   UYARI: Fonlama oranı çekilemedi ({symbol}): {e}. Fonlama 0 kabul edilecek."); return 0.0

def find_exact_trigger_price(symbol, df_4h, candle_index, position_type, rsi_ob, rsi_os, RSI_PERIOD=14):
    target_4h_candle = df_4h.iloc[candle_index]
    if candle_index < HISTORY_BUFFER: return None, None
    historical_closes_4h = df_4h['Close'].iloc[candle_index - HISTORY_BUFFER : candle_index].to_numpy()
    start_rsi = df_4h['rsi'].iloc[candle_index - 1]
    df_5m = get_binance_data(symbol, Client.KLINE_INTERVAL_5MINUTE, str(target_4h_candle['OpenTime']), limit=100)
    if df_5m is None or df_5m.empty: return None, None
    relevant_5m_candles = df_5m[(df_5m['OpenTime'] >= target_4h_candle['OpenTime']) & (df_5m['OpenTime'] < target_4h_candle['OpenTime'] + timedelta(hours=4))]
    if relevant_5m_candles.empty: return None, None
    rsi_in_loop = start_rsi
    for _, row in relevant_5m_candles.iterrows():
        low_price, high_price = row['Low'], row['High']
        rsi_at_low = talib.RSI(np.append(historical_closes_4h, low_price), timeperiod=RSI_PERIOD)[-1]
        rsi_at_high = talib.RSI(np.append(historical_closes_4h, high_price), timeperiod=RSI_PERIOD)[-1]
        if np.isnan(rsi_at_low) or np.isnan(rsi_at_high): continue
        if position_type == 'SHORT' and rsi_in_loop > rsi_os and rsi_at_low <= rsi_os:
            rsi_range = rsi_at_high - rsi_at_low
            price_range = high_price - low_price
            price_ratio = (rsi_os - rsi_at_low) / rsi_range if rsi_range > 0 else 0
            return low_price + (price_range * price_ratio), row['OpenTime']
        elif position_type == 'LONG' and rsi_in_loop < rsi_ob and rsi_at_high >= rsi_ob:
            rsi_range = rsi_at_high - rsi_at_low
            price_range = high_price - low_price
            price_ratio = (rsi_ob - rsi_at_low) / rsi_range if rsi_range > 0 else 0
            return low_price + (price_range * price_ratio), row['OpenTime']
        rsi_in_loop = talib.RSI(np.append(historical_closes_4h, row['Close']), timeperiod=RSI_PERIOD)[-1]
        if np.isnan(rsi_in_loop): break
    return None, None

def find_exit_tp_sl(symbol, start_candle_index, df_4h, entry_time, tp_price, sl_price, position_type):
    for i in range(start_candle_index, len(df_4h)):
        candle_4h = df_4h.iloc[i]
        start_time_5m = candle_4h['OpenTime']
        df_5m = get_binance_data(symbol, Client.KLINE_INTERVAL_5MINUTE, str(start_time_5m), limit=100)
        if df_5m is None or df_5m.empty: continue
        
        relevant_5m_candles = df_5m[df_5m['OpenTime'] >= entry_time]
        
        for _, row in relevant_5m_candles.iterrows():
            if position_type == 'LONG':
                if row['High'] >= tp_price:
                    return 'TP', tp_price, row['OpenTime'], i
                if row['Low'] <= sl_price:
                    return 'SL', sl_price, row['OpenTime'], i
            elif position_type == 'SHORT':
                if row['Low'] <= tp_price:
                    return 'TP', tp_price, row['OpenTime'], i
                if row['High'] >= sl_price:
                    return 'SL', sl_price, row['OpenTime'], i
        
        entry_time = candle_4h['OpenTime']

    last_candle = df_4h.iloc[-1]
    return 'TIMEOUT', last_candle['Close'], last_candle['OpenTime'], len(df_4h)


# ANA BACKTEST FONKSİYONU
def run_backtest(BACKTEST_DAYS, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT):
    start_date = datetime.now() - timedelta(days=BACKTEST_DAYS)
    start_date_str = start_date.strftime("%d %b, %Y")
    
    overall_results = {
        'total_pnl_percent': 0.0, 'long_wins': 0, 'long_losses': 0, 'short_wins': 0, 'short_losses': 0,
        'total_commission_percent': 0.0, 'total_funding_fee_percent': 0.0, 'timeout_closes': 0
    }
    coin_performance = {}

    for symbol in COIN_LIST:
        print(f"\n{'='*30}\n▶️ {symbol} için TP/SL Backtest Başlatılıyor...\n{'='*30}")
        df_4h = get_binance_data(symbol, Client.KLINE_INTERVAL_4HOUR, start_date_str, limit=1000)
        if df_4h is None or len(df_4h) < HISTORY_BUFFER + 2: continue
        df_4h['rsi'] = talib.RSI(df_4h['Close'], timeperiod=RSI_PERIOD)
        df_4h.dropna(inplace=True); df_4h.reset_index(drop=True, inplace=True)
        if len(df_4h) < HISTORY_BUFFER: continue

        coin_trades = []
        i = HISTORY_BUFFER
        while i < len(df_4h):
            current_candle = df_4h.iloc[i]
            previous_candle = df_4h.iloc[i-1]
            prev_rsi = previous_candle['rsi']
            
            signal_candidate = None
            base_closes_for_check = df_4h['Close'].iloc[i-HISTORY_BUFFER:i].to_numpy()
            potential_rsi_at_low = talib.RSI(np.append(base_closes_for_check, current_candle['Low']), RSI_PERIOD)[-1]
            potential_rsi_at_high = talib.RSI(np.append(base_closes_for_check, current_candle['High']), RSI_PERIOD)[-1]
            if prev_rsi > RSI_OVERSOLD and not np.isnan(potential_rsi_at_low) and potential_rsi_at_low <= RSI_OVERSOLD:
                signal_candidate = 'SHORT'
            elif prev_rsi < RSI_OVERBOUGHT and not np.isnan(potential_rsi_at_high) and potential_rsi_at_high >= RSI_OVERBOUGHT:
                signal_candidate = 'LONG'

            if signal_candidate:
                print(f"Sinyal Adayı Bulundu: {symbol} - Tarih: {current_candle['OpenTime']}")
                entry_price, entry_time = find_exact_trigger_price(symbol, df_4h, i, signal_candidate, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD)
                
                if entry_price and entry_time:
                    position_type = signal_candidate
                    if position_type == 'LONG':
                        tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
                        sl_price = entry_price * (1 - STOP_LOSS_PERCENT)
                    else:
                        tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
                        sl_price = entry_price * (1 + STOP_LOSS_PERCENT)
                    
                    print(f"🚀 YENİ POZİSYON AÇILDI: {symbol} | TÜR: {position_type} | GİRİŞ: {entry_price:.5f} | TP: {tp_price:.5f} | SL: {sl_price:.5f} | Tarih: {entry_time}")
                    
                    outcome, exit_price, exit_time, exit_candle_index = find_exit_tp_sl(symbol, i, df_4h, entry_time, tp_price, sl_price, position_type)
                    
                    if outcome == 'TP': gross_pnl_percent = TAKE_PROFIT_PERCENT
                    elif outcome == 'SL': gross_pnl_percent = -STOP_LOSS_PERCENT
                    else: gross_pnl_percent = (exit_price - entry_price) / entry_price if position_type == 'LONG' else (entry_price - exit_price) / entry_price
                    
                    commission_percent = 2 * COMMISSION_RATE
                    funding_fee_percent = get_funding_fee_percent(symbol, entry_time, exit_time, position_type)
                    net_pnl_percent = gross_pnl_percent - commission_percent + funding_fee_percent
                    
                    print(f"{'✅' if net_pnl_percent > 0 else '❌'} POZİSYON KAPANDI ({outcome}): {symbol} {position_type} pozisyonu {exit_price:.5f} fiyatından kapandı. Tarih: {exit_time}")
                    print(f"   PNL Detayları -> Brüt: {gross_pnl_percent*100:+.2f}%, Komisyon: {-commission_percent*100:.2f}%, Fonlama: {funding_fee_percent*100:+.4f}%, NET: {net_pnl_percent*100:+.2f}%")
                    
                    overall_results['total_pnl_percent'] += net_pnl_percent
                    overall_results['total_commission_percent'] += commission_percent
                    overall_results['total_funding_fee_percent'] += funding_fee_percent
                    if outcome == 'TIMEOUT': overall_results['timeout_closes'] += 1
                    
                    if outcome == 'TP':
                        if position_type == 'LONG': overall_results['long_wins'] += 1
                        else: overall_results['short_wins'] += 1
                    else: # SL veya TIMEOUT
                        if position_type == 'LONG': overall_results['long_losses'] += 1
                        else: overall_results['short_losses'] += 1

                    coin_trades.append({'pnl': net_pnl_percent})
                    i = exit_candle_index
                else:
                    print(f"Hassas giriş fiyatı bulunamadı. Mum atlanıyor.")
            
            i += 1
        
        if coin_trades:
            wins = sum(1 for t in coin_trades if t['pnl'] > 0)
            losses = len(coin_trades) - wins
            win_rate = (wins / len(coin_trades)) * 100 if len(coin_trades) > 0 else 0
            total_pnl = sum(t['pnl'] for t in coin_trades) * 100
            print(f"\n--- {symbol} Test Sonuçları ---")
            print(f"Toplam İşlem: {len(coin_trades)}, Kazanan: {wins}, Kaybeden: {losses}")
            print(f"Kazanma Oranı: {win_rate:.2f}%, Toplam PNL (Net): {total_pnl:.2f}%")
            coin_performance[symbol] = total_pnl
        else:
            print(f"\n--- {symbol} Test Sonuçları ---")
            print("Bu periyotta hiç işlem gerçekleşmedi.")
        time.sleep(0.5)

    # --- TAM VE DETAYLI RAPORLAMA ---
    report_lines = []
    separator = "="*40
    report_lines.append(separator)
    report_lines.append("🏆 SABİT TP/SL BACKTEST SONUÇLARI 🏆")
    report_lines.append(separator)
    
    report_lines.append("\n--- Test Parametreleri ---")
    report_lines.append(f"Test Edilen Gün Sayısı: {BACKTEST_DAYS}")
    report_lines.append(f"RSI Periyodu: {RSI_PERIOD}")
    report_lines.append(f"RSI Aşırı Satım (Oversold): {RSI_OVERSOLD}")
    report_lines.append(f"RSI Aşırı Alım (Overbought): {RSI_OVERBOUGHT}")
    report_lines.append(f"Take Profit Oranı: %{TAKE_PROFIT_PERCENT * 100:.2f}")
    report_lines.append(f"Stop Loss Oranı: %{STOP_LOSS_PERCENT * 100:.2f}")
    report_lines.append(f"Komisyon Oranı (İşlem Başına): %{COMMISSION_RATE * 100}")
    report_lines.append(f"Test Edilen Coin Sayısı: {len(COIN_LIST)}")
    report_lines.append("-" * 40)

    total_long_trades = overall_results['long_wins'] + overall_results['long_losses']
    long_win_rate = (overall_results['long_wins'] / total_long_trades * 100) if total_long_trades > 0 else 0
    report_lines.append("--- Long Pozisyonlar ---")
    report_lines.append(f"Toplam Long İşlem: {total_long_trades}")
    report_lines.append(f"  Kazanan Long (TP): {overall_results['long_wins']}")
    report_lines.append(f"  Kaybeden Long (SL): {overall_results['long_losses']}")
    report_lines.append(f"  Long Kazanma Oranı: {long_win_rate:.2f}%")
    report_lines.append("-" * 40)

    total_short_trades = overall_results['short_wins'] + overall_results['short_losses']
    short_win_rate = (overall_results['short_wins'] / total_short_trades * 100) if total_short_trades > 0 else 0
    report_lines.append("--- Short Pozisyonlar ---")
    report_lines.append(f"Toplam Short İşlem: {total_short_trades}")
    report_lines.append(f"  Kazanan Short (TP): {overall_results['short_wins']}")
    report_lines.append(f"  Kaybeden Short (SL): {overall_results['short_losses']}")
    report_lines.append(f"  Short Kazanma Oranı: {short_win_rate:.2f}%")
    report_lines.append("-" * 40)
    
    total_trades = total_long_trades + total_short_trades
    total_wins = overall_results['long_wins'] + overall_results['short_wins']
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    report_lines.append("--- Genel Toplam ---")
    report_lines.append(f"Toplam İşlem Sayısı: {total_trades}")
    report_lines.append(f"Genel Kazanma Oranı: {overall_win_rate:.2f}%")
    report_lines.append("--- Maliyet ve PNL Dökümü ---")
    report_lines.append(f"Toplam Komisyon Gideri: {overall_results['total_commission_percent'] * 100:.2f}%")
    report_lines.append(f"Toplam Fonlama Etkisi: {overall_results['total_funding_fee_percent'] * 100:+.4f}%")
    if overall_results['timeout_closes'] > 0:
        report_lines.append(f"Zaman Aşımı ile Kapanan: {overall_results['timeout_closes']} İşlem")
    report_lines.append(f"TOPLAM NET PNL: {overall_results['total_pnl_percent'] * 100:+.2f}%")
    report_lines.append("-" * 40)
    
    if coin_performance:
        report_lines.append("\n" + separator)
        report_lines.append("📈📉 COIN PERFORMANS SIRALAMASI (NET PNL) 📉📈")
        report_lines.append(separator)
        sorted_performance = sorted(coin_performance.items(), key=lambda item: item[1], reverse=True)
        for rank, (symbol, pnl) in enumerate(sorted_performance, 1):
            report_lines.append(f"{rank}. {symbol:<15} Toplam Net PNL: {pnl:+.2f}%")
            
    if total_trades == 0:
        report_lines.append("\nTest periyodunda belirtilen stratejiye uygun hiçbir işlem bulunamadı.")
        
    final_report_string = "\n".join(report_lines)
    print("\n\n" + final_report_string)
    
    try:
        if not os.path.exists('analizler'):
            os.makedirs('analizler')
        tp_str = int(TAKE_PROFIT_PERCENT * 1000)
        sl_str = int(STOP_LOSS_PERCENT * 1000)
        pnl_str = f"{overall_results['total_pnl_percent'] * 100:.2f}".replace('.', '_').replace('-', 'm')
        filename = (f"analizler/tpsl_{BACKTEST_DAYS}d_rsi{RSI_PERIOD}-{RSI_OVERSOLD}-{RSI_OVERBOUGHT}_tp{tp_str}_sl{sl_str}_pnl{pnl_str}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_report_string)
        print(f"\n✅ Rapor başarıyla '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"\n❌ Rapor dosyaya kaydedilirken bir hata oluştu: {e}")

# --- OPTİMİZASYON YÖNETİCİSİ ---
def start_optimization():
    BACKTEST_DAYS = 150
    RSI_PERIOD = 14
    RSI_OVERSOLD_LIST = [35]
    RSI_OVERBOUGHT_LIST = [65]

    tp_range = np.arange(0.045, 0.046, 0.005)
    sl_range = np.arange(0.01, 0.011, 0.005)

    total_combinations = 0
    for rsi_os in RSI_OVERSOLD_LIST:
        for rsi_ob in RSI_OVERBOUGHT_LIST:
            if (rsi_ob + rsi_os) != 100: continue
            for tp in tp_range:
                for sl in sl_range:
                    if tp < sl: continue
                    total_combinations += 1
    
    print(f"Sabit TP/SL Optimizasyonu Başlatılıyor... Toplam {total_combinations} farklı parametre kombinasyonu test edilecek.")
    
    test_counter = 0
    for rsi_os in RSI_OVERSOLD_LIST:
        for rsi_ob in RSI_OVERBOUGHT_LIST:
            if (rsi_ob + rsi_os) != 100: continue
            for tp in tp_range:
                for sl in sl_range:
                    if tp < sl: continue
                    test_counter += 1
                    tp = round(tp, 4)
                    sl = round(sl, 4)
                    
                    print(f"\n\n{'#'*60}")
                    print(f"### TEST {test_counter} / {total_combinations} ###")
                    print(f"Parametreler: RSI={RSI_PERIOD}/{rsi_os}/{rsi_ob} | TP=%{tp*100:.2f} | SL=%{sl*100:.2f}")
                    print(f"{'#'*60}\n")
                    
                    run_backtest(
                        BACKTEST_DAYS=BACKTEST_DAYS,
                        RSI_PERIOD=RSI_PERIOD,
                        RSI_OVERSOLD=rsi_os,
                        RSI_OVERBOUGHT=rsi_ob,
                        TAKE_PROFIT_PERCENT=tp,
                        STOP_LOSS_PERCENT=sl 
                    )
    print("\n\nOptimizasyon süreci tamamlandı!")


if __name__ == "__main__":
    start_optimization()