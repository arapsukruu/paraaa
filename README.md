# Quant Lab — Kripto Vadeli Otomatik Strateji Motoru (`CORE_OI_CASCADE`)

Klasik 5dk/15dk RSI/MACD kalıpları yerine **gerçek borsa tasfiye (likidasyon) akışı**, **Open Interest (OI) erimesi**, **canlı Parite DNA'sı (`Variance Ratio`)** ve **Makro BTC 24s Rejim Kalkanı** üzerine kurulu araştırma ve canlı Paper-Trading altyapısı.

## 1. Canlı Otomatik Paper-Trading Motorunu Çalıştırma

API anahtarı veya gerçek para **gerektirmez** (Gate.io halka açık USDT-M vadeli/spot/tasfiye/OI akışını okur):

```powershell
# Tek seferlik anlık piyasa taraması ve Canlı Radar Dashboard güncellemesi:
python quant_lab/live_paper_engine.py --once

# 7/24 Sürekli Otomatik Paper-Trading (her 5dk bar kapanışında otomatik tarar):
python quant_lab/live_paper_engine.py --loop --capital 10000

# Sanal hesabı sıfırlamak için:
python quant_lab/live_paper_engine.py --once --capital 10000 --reset

# Yerel çevrimdışı tekrar testi (internet gerekmez):
python quant_lab/live_paper_engine.py --offline-test
```

Canlı durum panelini (her 60 saniyede otomatik yenilenir) tarayıcıda açmak için:
- `quant_lab/live_state/live_dashboard.html`

İşlem kayıtları ve durum dosyaları:
- `quant_lab/live_state/portfolio_state.json` — Anlık bakiye, açık pozisyonlar, devre kesici durumu
- `quant_lab/live_state/trades_log.csv` — Kapanan tüm sanal işlemlerin kalıcı günlüğü
- `quant_lab/live_state/signals_log.jsonl` — Açılan ve risk filtresine takılan tüm aday sinyaller

## 2. Araştırma & Backtest Tekrarı

```powershell
# 1) Gate.io 32 günlük 16 parite mikroyapı verisini yenile
python quant_lab/data_collector.py --days 32 --workers 4

# 2) 1. Tur: 4 Metod x 3 Vade Walk-Forward Turnuvası
python quant_lab/run_tournament.py

# 3) 2. Tur & Anti-İllüzyon / Gecikme Stres Testi
python quant_lab/run_round2_adaptive.py
python quant_lab/audit_round2.py
python quant_lab/stress_test_cascade.py

# 4) 3. Tur: Güçlendirilmiş Çekirdek Motor (Hibrit Maliyet OOS Doğrulama)
python quant_lab/run_round3_hardened.py
```

## 3. Çekirdek Strateji Kuralları (`CORE_OI_CASCADE`)
1. **Gerçek Tasfiye Akışı:** `|real_liq_z| >= 1.4` (veya `|liq_cascade_score| >= 1.9`) + `|real_liq_z| <= 4.5` (son zirve kapitülasyon iğnesinden kaçınma).
2. **Gerçek Kaldıraç Erimesi:** 15 dakikalık Open Interest değişimi `<= -0.55%` (yutulan sahte iğneleri eler).
3. **Canlı Parite DNA'sı & Volatilite Tabanı:** `vr_15 >= 0.68` (aşırı yatay pariteyi eler) ve `vol_240m >= 12 bps`.
4. **Makro BTC 24s Trend Kalkanı:** Güçlü 24s makro trende ters yönde (`> 1.2 sigma`) giriş engellenir.
5. **Risk & Maliyet Yönetimi:** Volatiliteye göre eşit risk boyutlandırması (`%0.55` risk/işlem), maks 6 eş zamanlı pozisyon, maks `1.35x` toplam maruziyet, UTC günlük `%-2.5` devre kesici, gerçekçi Hibrit maliyet (`8 bps` giriş + TP'de `2 bps` Maker / SL'de `7 bps` Taker + gerçek funding).
