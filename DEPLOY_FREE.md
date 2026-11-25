# Ücretsiz 7/24 Telegram Bildirimli Sunucu Kurulum Rehberi

## Adım 1 — 1 Dakikada Telegram Botu ve Chat ID Alma
1. Telegram'da **`@BotFather`** aç → `/newbot` yaz → bot adını ve `..._bot` kullanıcı adını belirle → verilen **`TELEGRAM_BOT_TOKEN`** değerini kopyala.
2. Oluşturduğun kendi botunu aç ve **`Başlat (Start)`** butonuna bas.
3. Telegram'da **`@userinfobot`** aç → `/start` yaz → dönen sayısal **`Id`** senin **`TELEGRAM_CHAT_ID`** değerindir.

---

## Adım 2 — Önce Kendi Bilgisayarında Anında Test Et
1. `quant_lab/.env.example` dosyasını `quant_lab/.env` olarak kopyala (veya yeni `quant_lab/.env` oluştur) ve şu iki satırı yapıştır:
   ```env
   TELEGRAM_BOT_TOKEN=BURAYA_TOKEN
   TELEGRAM_CHAT_ID=BURAYA_CHAT_ID
   ```
2. Tarayıcıda **http://127.0.0.1:8787** panelini açıp sağ üstteki **`Telegram: ... (Test)`** butonuna tıkla (sunucuyu yeniden başlatmana bile gerek yok, `.env` dosyasını anında okur!).
3. Telefonuna test bildirimi geldiği an entegrasyon %100 tamamdır.

---

## Adım 3 — PC Kapalıyken 7/24 Çalışması İçin Ücretsiz Bulut Kurulumu (Kredi Kartsız)

### Seçenek A: Render.com + UptimeRobot (En Kolay, 3 Dakika)
1. GitHub'da **Private (Gizli)** bir repo oluştur ve yalnızca **`quant_lab/`** klasörünün içindekileri yükle (`.gitignore` sayesinde `.env` yüklenmez).
2. **https://render.com** → GitHub ile ücretsiz giriş → **New + → Web Service** → reponu seç.
   * **Runtime:** `Docker` (otomatik `Dockerfile`'ı algılar)
   * **Instance Type:** `Free`
   * **Health Check Path:** `/health`
   * **Environment Variables** bölümüne şu 3 anahtarı ekle:
     * `TELEGRAM_BOT_TOKEN` = *(BotFather tokenın)*
     * `TELEGRAM_CHAT_ID` = *(Chat ID'n)*
     * `DEMO_CAPITAL` = `5000`
3. **Create Web Service** tıkla (~2 dk içinde canlıya geçer, sana `https://....onrender.com` linki verir ve telefonuna **"🚀 CORE_OI_CASCADE Bot Çevrimiçi"** mesajı düşer).
4. **Uyumaması İçin (Önemli):** Ücretsiz **https://cron-job.org** veya **https://uptimerobot.com** hesabı aç → her **5 dakikada bir** `https://SENIN-BOTUN.onrender.com/health` adresine GET isteği atacak bir monitor ekle. Artık PC kapalıyken de 7/24 uyanık kalır ve canlı HTML paneline telefondan bile girebilirsin!

### Seçenek B: Hugging Face Spaces (Kredi Kartsız, 16 GB RAM)
1. **https://huggingface.co/new-space** → Space adı ver → **SDK: Docker (Blank)** → **Visibility: Private** seç → **Create Space**.
2. `quant_lab/` içindeki kod dosyalarını (`demo_bot_server.py`, `core_strategy.py`, `features.py`, `data_collector.py`, `config.py`, `telegram_notifier.py`, `engine/`, `templates/`, `requirements.txt`, `Dockerfile`) yükle.
3. Space **Settings → Variables and secrets → New secret** kısmından `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` ekle.
4. Space ayağa kalkınca **cron-job.org** üzerinden saatte 1 kez Space URL'ine ping atarak 48 saatlik uykuyu engelle.
