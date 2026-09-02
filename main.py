import os
import io
import re
import base64
import json
import time
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", category=UserWarning)

# 12 saatlik cache
chart_cache = TTLCache(maxsize=50, ttl=43200)
cache_lock = threading.Lock()

_symbol_locks = defaultdict(Lock)
_locks_lock = Lock()


def get_symbol_lock(symbol: str) -> Lock:
    with _locks_lock:
        return _symbol_locks[symbol]


yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
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

GEMINI_MODELS = [
    "gemini-3.8-flash",       # En yeni + en güçlü
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",  # Yoğunlukta hızlı kaçış rotası
]

# Sadece bağımsız "nan" / "NaN" (Türkçe kelime içi eşleşme yok)
NAN_RE = re.compile(r"(?<![a-zçğıöşüA-ZÇĞİÖŞÜ])nan(?![a-zçğıöşüA-ZÇĞİÖŞÜ])", re.IGNORECASE)


def get_current_usd_try() -> float:
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return float(data["rates"]["TRY"])
    except Exception:
        return 48.00


def generate_with_retry(contents, max_attempts: int = 4):
    last_err = None
    for attempt in range(max_attempts):
        model = GEMINI_MODELS[min(attempt, len(GEMINI_MODELS) - 1)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(
                x in msg
                for x in (
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted",
                    "high demand",
                    "overloaded",
                    "try again",
                )
            )
            if retryable and attempt < max_attempts - 1:
                time.sleep(min(1.5 * (2 ** attempt), 20))
                continue
            raise
    raise last_err


@app.get("/analiz")
def analiz_et(hisse: str = Query("BRSAN"), x_api_key: str = Header(None)):
    if not API_SECRET_KEY or x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Erişim reddedildi. Geçersiz API anahtarı.")

    hisse_kodu = hisse.upper().strip()
    if not hisse_kodu.endswith(".IS"):
        hisse_kodu += ".IS"

    with cache_lock:
        if hisse_kodu in chart_cache:
            return chart_cache[hisse_kodu]

    lock = get_symbol_lock(hisse_kodu)
    with lock:
        with cache_lock:
            if hisse_kodu in chart_cache:
                return chart_cache[hisse_kodu]

        try:
            ticker = yf.Ticker(hisse_kodu, session=yf_session)
            df = ticker.history(period="3y", interval="1wk")
        except Exception as e:
            return {"image": None, "rapor": f"Veri çekme hatası (Yahoo Limit): {str(e)}"}

        if df is None or df.empty:
            return {
                "image": None,
                "rapor": f"HATA: '{hisse_kodu}' için veri bulunamadı! Sembolü kontrol edin.",
            }

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                return {
                    "image": None,
                    "rapor": f"HATA: '{hisse_kodu}' veri kolonları eksik ({col}).",
                }

        df = df.dropna(subset=["Close", "Open", "High", "Low"])
        if df.empty:
            return {
                "image": None,
                "rapor": f"HATA: '{hisse_kodu}' için geçerli mum verisi yok.",
            }

        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA22"] = df["Close"].rolling(window=22).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()

        add_plots = [
            mpf.make_addplot(df["MA5"], color="cyan", width=1.2),
            mpf.make_addplot(df["MA22"], color="orange", width=1.2),
            mpf.make_addplot(df["MA50"], color="lime", width=1.2),
        ]

        img_io = io.BytesIO()
        try:
            fig, axes = mpf.plot(
                df,
                type="candle",
                style="nightclouds",
                title=f"\n{hisse_kodu} Haftalik Teknik Grafik (TL)",
                ylabel="Fiyat (TL)",
                volume=True,
                addplot=add_plots,
                panel_ratios=(3, 1),
                figratio=(14, 8),
                figscale=1.2,
                returnfig=True,
            )
            fig.savefig(img_io, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            img_io.seek(0)
            plt_img = Image.open(img_io).convert("RGB")
            plt.close(fig)
            plt.close("all")
        except Exception as e:
            try:
                plt.close("all")
            except Exception:
                pass
            return {"image": None, "rapor": f"Grafik çizilirken hata oluştu: {str(e)}"}

        guncel_kur = get_current_usd_try()
        guncel_fiyat_tl = float(df["Close"].iloc[-1])

        if pd.isna(guncel_fiyat_tl) or guncel_fiyat_tl <= 0:
            return {
                "image": None,
                "rapor": f"HATA: '{hisse_kodu}' son kapanış geçersiz.",
            }

        guncel_fiyat_usd = guncel_fiyat_tl / guncel_kur

        sistem_istemi = rf"""
    Sen kıdemli bir teknik analist ve algoritmik trade uzmanısın. Özel uzmanlık alanın Elliott Dalga Teorisi (EDT).
    Piyasadaki anlık Dolar/TL kuru **1 USD = {guncel_kur:.2f} TL** seviyesindedir. 
    İncelediğin **{hisse_kodu}** hissesinin Python tarafından bizzat doğrulanan anlık güncel fiyatı: **{guncel_fiyat_tl:.2f} TL** (yaklaşık **${guncel_fiyat_usd:.2f} USD**) kadardır. Analizini ve fiyat hedeflerini MUTLAKA bu gerçek güncel fiyatı baz alarak yap.
    === KRİTİK KURALLAR ===
1. Grafik ekseni TÜRK LİRASI (TL). Sayıları ASLA USD sanma.
2. Güncel fiyatı grafikten okuma; yalnızca {guncel_fiyat_tl:.2f} TL kullan.
3. Seviyeler: XXX.XX TL ($Y.YY USD) — USD = TL / {guncel_kur:.2f}
4. Asla "nan", "NaN", "undefined", "null" yazma.
5. Mevcut fiyat satırında birebir: {guncel_fiyat_tl:.2f} TL (${guncel_fiyat_usd:.2f} USD)
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

        basarili_mi = False
        try:
            response = generate_with_retry([sistem_istemi, plt_img], max_attempts=4)
            analiz_metni = (response.text or "").strip()
            if not analiz_metni:
                analiz_metni = "Yapay zeka boş yanıt döndü. Tekrar deneyin."
                basarili_mi = False
            elif NAN_RE.search(analiz_metni):
                # Gerçek NaN sızıntısı — cache'leme
                analiz_metni = (
                    analiz_metni
                    + "\n\n[Sistem] Raporda geçersiz NaN tespit edildi; cache'lenmedi. Tekrar deneyin."
                )
                basarili_mi = False
            else:
                basarili_mi = True
        except Exception as e:
            analiz_metni = f"Yapay zeka analiz raporu oluşturulurken hata oluştu: {str(e)}"
            basarili_mi = False

        # Görseli yeniden encode (RGB PNG)
        out = io.BytesIO()
        plt_img.save(out, format="PNG", optimize=True)
        base64_img = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")

        result = {
            "image": base64_img,
            "rapor": analiz_metni,
        }

        if basarili_mi:
            with cache_lock:
                chart_cache[hisse_kodu] = result

        return result
