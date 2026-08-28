import os
import io
import base64
import json
import urllib.request
import threading
from collections import defaultdict
from threading import Lock
import requests
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PIL import Image
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from cachetools import TTLCache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 12 saatlik cache
chart_cache = TTLCache(maxsize=50, ttl=43200)
cache_lock = threading.Lock()

# Hisse bazlı kilitler (thundering herd önleme)
_symbol_locks = defaultdict(Lock)
_locks_lock = Lock()   # _symbol_locks dict'ini korumak için

def get_symbol_lock(symbol: str) -> Lock:
    with _locks_lock:
        return _symbol_locks[symbol]

# Yahoo Finance için daha stabil session
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
client = genai.Client(api_key=API_KEY)

def get_current_usd_try():
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data['rates']['TRY']
    except Exception:
        return 48.00

@app.get("/analiz")
def analiz_et(hisse: str = Query("BRSAN"), x_api_key: str = Header(None)):
    if not API_SECRET_KEY or x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Erişim reddedildi. Geçersiz API anahtarı.")
    
    hisse_kodu = hisse.upper()
    if not hisse_kodu.endswith(".IS"):
        hisse_kodu += ".IS"

    # 1. Hızlı Cache Kontrolü
    with cache_lock:
        if hisse_kodu in chart_cache:
            return chart_cache[hisse_kodu]

    # 2. Eşzamanlı İstek Engelleme (Hisseye özel kilit)
    lock = get_symbol_lock(hisse_kodu)
    with lock:
        # Double-check
        with cache_lock:
            if hisse_kodu in chart_cache:
                return chart_cache[hisse_kodu]

        # 3. Custom Session ile Veri Çekme
        try:
            ticker = yf.Ticker(hisse_kodu, session=yf_session)
            df = ticker.history(period="3y", interval="1wk")
        except Exception as e:
            return {"image": None, "rapor": f"Veri çekme hatası (Yahoo Limit): {str(e)}"}

        if df.empty:
            return {"image": None, "rapor": f"HATA: '{hisse_kodu}' için veri bulunamadı! Sembolü kontrol edin."}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA22'] = df['Close'].rolling(window=22).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        add_plots = [
            mpf.make_addplot(df['MA5'], color='cyan', width=1),
            mpf.make_addplot(df['MA22'], color='orange', width=1),
            mpf.make_addplot(df['MA50'], color='green', width=1),
        ]

        img_io = io.BytesIO()
        try:
            fig, axes = mpf.plot(
                df, 
                type='candle', 
                style='nightclouds', 
                title=f"\n{hisse_kodu} Haftalik Teknik Grafik",
                ylabel='Fiyat',
                volume=True,
                addplot=add_plots,
                panel_ratios=(3,1),
                figratio=(12,8),
                returnfig=True
            )
            fig.savefig(img_io, format='png', dpi=300, bbox_inches='tight')
            img_io.seek(0)
            plt_img = Image.open(img_io)
            plt.close(fig)
        except Exception as e:
            return {"image": None, "rapor": f"Grafik çizilirken hata oluştu: {str(e)}"}

        guncel_kur = get_current_usd_try()
        guncel_fiyat_tl = float(df['Close'].iloc[-1])
        guncel_fiyat_usd = guncel_fiyat_tl / guncel_kur

        sistem_istemi = rf"""
    Sen kıdemli bir teknik analist ve algoritmik trade uzmanısın. Özel uzmanlık alanın Elliott Dalga Teorisi (EDT).
    Piyasadaki anlık Dolar/TL kuru **1 USD = {guncel_kur:.2f} TL** seviyesindedir. 
    İncelediğin **{hisse_kodu}** hissesinin Python tarafından bizzat doğrulanan anlık güncel fiyatı: **{guncel_fiyat_tl:.2f} TL** (yaklaşık **${guncel_fiyat_usd:.2f} USD**) kadardır. Analizini ve fiyat hedeflerini MUTLAKA bu gerçek güncel fiyatı baz alarak yap.

    Sana verdiğim teknik grafik görüntüsünü ve yukarıdaki gerçek fiyat verisini birlikte değerlendirerek şu adımları harfiyen yerine getir:

    1. Grafikteki majör dalgaları (1-2-3-4-5) veya düzeltme dalgalarını (A-B-C) detaylıca tespit et.
    2. Tespit ettiğin fiyat seviyelerini hem DOLAR (USD) hem de canlı kur ({guncel_kur:.2f} TL) üzerinden LİRA (TL) cinsinden parantez içinde belirt.
    3. Grafikteki fiyatlara göre hem USD hem TL uyumlu ASCII dalga şemasını mutlaka çiz:
    
    [Fiyat USD / TL] Eski Zirve           [Fiyat USD / TL] Çift Tepe / Wave B Testi
               /\                        /\
              /  \                      /  \   <-- Düzeltme / Reddedilme (Wave C veya 4)
             /    \                    /    \
            /      \                  /      \       [Fiyat USD / TL] Olası Dip
           /        \                /        \_____/
    [Fiyat USD / TL] \              /
                      \____________/
                       [Fiyat USD / TL] Dip

    4. Grafikteki mevcut durumu, hacim uyumunu ve hareketli ortalamaları (MA5, MA22, MA50) değerlendirerek ne zaman düzeltmeye/yükselişe geçeceğini olasılık yüzdeleri vererek her iki para birimi bazında açıkla.
    5. **ÖNEMLİ:** Kesinlikle karmaşık Markdown tabloları KULLANMA. Aşağıdaki formatı birebir takip ederek temiz, alt alta maddeler halinde "SEVİYE VE HEDEF LİSTESİ" sun:

    - **Stop-Loss / Stop Seviyesi:** $X.XX USD (XXX TL) - Açıklama metni...
    - **Kritik Destek (Ana Taban):** $X.XX USD (XXX TL) - Açıklama metni...
    - **Ara Destek (Tetiklenme):** $X.XX USD (XXX TL) - Açıklama metni...
    - **Mevcut Fiyat:** ${guncel_fiyat_usd:.2f} USD ({guncel_fiyat_tl:.2f} TL) - Anlık Grafik Fiyatı
    - **İlk Ara Direnç:** $X.XX USD (XXX TL) - Açıklama metni...
    - **Ana Hedef (Wave 5 Target):** $X.XX USD (XXX TL) - Açıklama metni...
    - **Tarihi Majör Direnç:** $X.XX USD (XXX TL) - Açıklama metni...
    """

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[sistem_istemi, plt_img]
            )
            analiz_metni = response.text
            basarili_mi = True
        except Exception as e:
            analiz_metni = f"Yapay zeka analiz raporu oluşturulurken hata oluştu: {str(e)}"
            basarili_mi = False

        img_io.seek(0)
        base64_img = "data:image/png;base64," + base64.b64encode(img_io.read()).decode('utf-8')

        result = {
            "image": base64_img,
            "rapor": analiz_metni
        }

        if basarili_mi:
            with cache_lock:
                chart_cache[hisse_kodu] = result

        return result
