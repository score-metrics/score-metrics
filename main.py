from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import poisson

app = FastAPI(title="Score Metrics API", version="1.0")

# Configurar CORS para permitir peticiones desde tu frontend en Vercel y desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_xg: float  # Goles esperados del local (ej. 1.65)
    away_xg: float  # Goles esperados del visitante (ej. 1.10)

@app.get("/")
def read_root():
    return {"status": "online", "service": "Score Metrics Engine"}

@app.post("/api/predict")
def calculate_match_probabilities(data: MatchInput):
    """
    Calcula las probabilidades de victoria local, empate y visitante
    usando la distribución de Poisson basada en xG.
    """
    max_goals = 10
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    # Calcular matriz de probabilidades de goles (Poisson)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_home = poisson.pmf(h, data.home_xg)
            p_away = poisson.pmf(a, data.away_xg)
            match_prob = p_home * p_away

            if h > a:
                home_win_prob += match_prob
            elif h == a:
                draw_prob += match_prob
            else:
                away_win_prob += match_prob

    # Normalizar porcentajes
    total = home_win_prob + draw_prob + away_win_prob
    h_pct = round((home_win_prob / total) * 100, 1)
    d_pct = round((draw_prob / total) * 100, 1)
    a_pct = round((away_win_prob / total) * 100, 1)

    return {
        "match": f"{data.home_team} vs {data.away_team}",
        "probabilities": {
            "home_win": f"{h_pct}%",
            "draw": f"{d_pct}%",
            "away_win": f"{a_pct}%"
        },
        "raw_values": {
            "home_win": h_pct,
            "draw": d_pct,
            "away_win": a_pct
        }
    }
