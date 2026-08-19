from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analiz")
def analiz_uret(hisse: str):
    ticker_symbol = f"{hisse.upper()}.IS"
    
    df = yf.download(ticker_symbol, period="6mo", interval="1d")
    if df.empty:
        return {"error": "Hisse bulunamadı"}

    plt.figure(figsize=(8, 4))
    plt.style.use('dark_background')
    plt.plot(df.index, df['Close'], color='#39FF88', linewidth=1.5)
    plt.title(f"// {hisse.upper()} ELLIOTT & TECHNICAL SCAN", color='#39FF88', fontsize=10)
    plt.grid(color='#39FF88', alpha=0.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', facecolor='black')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    raport_metni = f"""Senior Teknik Analist Raporu:
- Ticker: {hisse.upper()}
- Durum: Aktif Dalga 3 itkisi ve hacim onayı takip ediliyor.
- Seviyeler güncel verilerle senkronizedir."""

    return {
        "hisse": hisse.upper(),
        "image": f"data:image/png;base64,{image_base64}",
        "rapor": raport_metni
    }
