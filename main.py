import os
import io
import base64
import json
import urllib.request
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PIL import Image
import yfinance as yf
import mplfinance as mpf
import pandas as pd

app = FastAPI()

# Frontend ile CORS problemi yaşamamak için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def get_current_usd_try():
    """Anlık USD/TRY kurunu çeker."""
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
    # --- GÜVENLİK KONTROLÜ ---
    # Dışarıdan izinsiz botların istek atmasını engeller
    if not API_SECRET_KEY or x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Erişim reddedildi. Geçersiz API anahtarı.")    hisse_kodu = hisse.upper()
        
    if not hisse_kodu.endswith(".IS"):
        hisse_kodu += ".IS"
        
    # 1. Yahoo Finance üzerinden haftalık verileri çekme
    df = yf.download(hisse_kodu, period="3y", interval="1wk", progress=False)
    if df.empty:
        return {"image": None, "rapor": f"HATA: '{hisse_kodu}' için veri bulunamadı! Sembolü kontrol edin."}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Hareketli Ortalama (MA) İndikatörleri
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
    except Exception as e:
        return {"image": None, "rapor": f"Grafik çizilirken hata oluştu: {str(e)}"}

    guncel_kur = get_current_usd_try()

    # Profesyonel Kıdemli Analist ve EDT Promptu
    sistem_istemi = rf"""
    Sen kıdemli bir teknik analist ve algoritmik trade uzmanısın. Özel uzmanlık alanın Elliott Dalga Teorisi (EDT).
    Piyasadaki anlık Dolar/TL kuru **1 USD = {guncel_kur:.2f} TL** seviyesindedir. 
    Sana vereceğim otomatik oluşturulmuş fiyat grafiği görüntüsünü inceleyerek şu adımları harfiyen yerine getir:

    1. Grafikteki majör dalgaları (1-2-3-4-5) veya düzeltme dalgalarını (A-B-C) detaylıca tespit et.
    2. Tespit ettiğin fiyat seviyelerini (tepe, dip, kırılım) hem DOLAR (USD) hem de yukarıda verilen canlı kur ({guncel_kur:.2f} TL) üzerinden LİRA (TL) cinsinden parantez içinde belirt.
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
    5. Olası dip, tepe, tetiklenme ve ana hedef seviyelerini hem USD hem TL içeren temiz bir Markdown tablosu halinde sun.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[sistem_istemi, plt_img]
        )
        analiz_metni = response.text
    except Exception as e:
        analiz_metni = f"Yapay zeka analiz raporu oluşturulurken hata oluştu: {str(e)}"

    img_io.seek(0)
    base64_img = "data:image/png;base64," + base64.b64encode(img_io.read()).decode('utf-8')

    return {
        "image": base64_img,
        "rapor": analiz_metni
    }
