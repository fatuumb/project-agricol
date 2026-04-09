"""
BayAI — Dashboard Streamlit v3.0
Données réelles Open-Meteo | 2010 → semaine -1 dynamique
Run : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import requests
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score,
    f1_score, accuracy_score,
    classification_report, ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BayAI — Smart Agriculture",
    page_icon="🌾",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e4ede4;
    border-radius: 14px;
    padding: 1rem 1.1rem 0.85rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #1D9E75);
    border-radius: 14px 14px 0 0;
}
.kpi-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #7a8a7a;
    margin-bottom: 5px;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: #1a2e1a;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 10.5px;
    color: #9aaa9a;
    margin-top: 4px;
    line-height: 1.4;
}

.section-title {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #3d5c3d;
    padding-bottom: 7px;
    border-bottom: 1.5px solid #e0ebe0;
    margin-bottom: 14px;
    margin-top: 4px;
}

.alert-box {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 13px;
    border-left: 4px solid;
    line-height: 1.5;
}
.alert-critical { background:#fff1f1; color:#7b1d1d; border-color:#e53e3e; }
.alert-warning  { background:#fffbeb; color:#78350f; border-color:#f59e0b; }
.alert-info     { background:#eff6ff; color:#1e3a5f; border-color:#3b82f6; }
.alert-ok       { background:#f0fdf4; color:#14532d; border-color:#22c55e; }

.sidebar-pill {
    display: inline-block;
    background: #e8f5ee;
    color: #0f6e50;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11.5px;
    font-weight: 600;
}
.sidebar-sep {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #a0b0a0;
    margin: 14px 0 5px;
}

.bayai-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0 14px;
}
.bayai-logo  { font-size: 34px; line-height: 1; }
.bayai-title { font-size: 24px; font-weight: 600; color: #1a2e1a; letter-spacing: -.02em; }
.bayai-sub   { font-size: 12px; color: #7a8a7a; margin-top: 2px; }
.header-divider { border: none; border-top: 1.5px solid #e0ebe0; margin: 0 0 18px; }

/* ── Sliders ── */
div[data-baseweb='slider'] [role='slider'] {
    background-color: #1D9E75 !important;
    border-color: #1D9E75 !important;
}
div[data-baseweb='slider'] [data-testid='stSliderTrackFill'] {
    background-color: #1D9E75 !important;
}
div[data-testid='stSliderThumbValue'] {
    color: #1D9E75 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
/* ── Boutons ── */
div[data-testid='stButton'] button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
/* ── Masquer la toolbar Streamlit (Deploy, etc.) ── */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
div[data-testid="stToolbar"]         { display: none !important; }
div[data-testid="stDecoration"]      { display: none !important; }
div[data-testid="stStatusWidget"]    { display: none !important; }
div[data-testid="manage-app-button"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Villes disponibles
# ─────────────────────────────────────────────────────────
CITIES: dict[str, tuple[float, float]] = {
    "🇸🇳 Dakar, Sénégal":          (14.69, -17.44),
    "🇲🇱 Bamako, Mali":             (12.65,  -8.00),
    "🇨🇮 Abidjan, Côte d'Ivoire":  ( 5.35,  -4.00),
    "🇬🇭 Accra, Ghana":             ( 5.55,  -0.20),
    "🇳🇬 Lagos, Nigeria":           ( 6.52,   3.38),
    "🇰🇪 Nairobi, Kenya":           (-1.29,  36.82),
    "🇪🇹 Addis-Abeba, Éthiopie":   ( 9.03,  38.74),
    "🇲🇦 Marrakech, Maroc":        (31.63,  -8.01),
}

# ─────────────────────────────────────────────────────────
# Date de fin dynamique : lundi de la semaine précédente
# ─────────────────────────────────────────────────────────
def get_end_date() -> str:
    today = date.today()
    last_monday = today - timedelta(days=today.weekday() + 7)
    return last_monday.strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────
# Chargement données Open-Meteo
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# Labellisation du risque — calibration locale par percentiles
# ─────────────────────────────────────────────────────────
def label_risk(df: pd.DataFrame) -> pd.Series:
    """
    Calibration adaptative du risque de sécheresse par percentiles locaux.

    Principe : détecter les ANOMALIES par rapport à la norme climatique
    de chaque région et de chaque mois. Un sol à 5% d'humidité est normal
    à Dakar en saison sèche — mais anormal en hivernage.

    3 signaux combinés (calibrés sur validation historique) :
      A — Chaleur extrême locale  : temp_max > P85 du mois
      B — Sol anormalement sec    : soil_moisture < P20 du mois
      C — Déficit hydrique cumulé : déficit 14j glissant > P75 du mois

    Risque = True si au moins 2 signaux sur 3 sont actifs.
    → Taux cible : 10-15% sur longue période, avec pics sur années
      de sécheresse documentées (Sahel 2011, 2017, 2022).
    """
    d = df.copy()

    # Colonne _month nécessaire pour les percentiles mensuels
    if "date" not in d.columns:
        d["date"] = pd.date_range("2010-01-01", periods=len(d), freq="D")
    d["_month"] = pd.to_datetime(d["date"]).dt.month

    # Signal A — Chaleur extrême locale (P85 mensuel)
    # Seuil relatif : un 35°C en janvier à Dakar = anomalie, pas en juillet
    p85_temp = d.groupby("_month")["temp_max"].transform(
        lambda x: x.quantile(0.85)
    )

    # Signal B — Sol anormalement sec (P20 mensuel)
    # Détecte un dessèchement inhabituellement bas pour ce mois
    p20_soil = d.groupby("_month")["soil_moisture"].transform(
        lambda x: x.quantile(0.20)
    )

    # Signal C — Déficit hydrique cumulé sur 14 jours (P75 mensuel)
    # Le stress hydrique s'accumule — un jour sec ne suffit pas,
    # mais 2 semaines de déficit cumulé indiquent une vraie sécheresse
    d["_deficit"] = (d["evapotranspiration"] - d["rainfall"]).clip(lower=0)
    d["_deficit_14d"] = (
        d["_deficit"].rolling(14, min_periods=5).mean().fillna(d["_deficit"])
    )
    p75_deficit = d.groupby("_month")["_deficit_14d"].transform(
        lambda x: x.quantile(0.75)
    )

    # Signaux binaires
    signal_A = (d["temp_max"]      > p85_temp).astype(int)
    signal_B = (d["soil_moisture"] < p20_soil).astype(int)
    signal_C = (d["_deficit_14d"]  > p75_deficit).astype(int)

    # Risque si au moins 2 signaux sur 3 actifs simultanément
    risk = ((signal_A + signal_B + signal_C) >= 2).astype(int)

    return risk


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_weather(lat: float, lon: float,
                       start_date: str = "2010-01-01",
                       end_date: str | None = None) -> pd.DataFrame:
    if end_date is None:
        end_date = get_end_date()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "soil_moisture_0_to_7cm_mean",
            "et0_fao_evapotranspiration",
            "wind_speed_10m_max",
        ],
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        raw = r.json().get("daily", {})
        df = pd.DataFrame(raw).rename(columns={
            "time":                         "date",
            "temperature_2m_max":           "temp_max",
            "temperature_2m_min":           "temp_min",
            "precipitation_sum":            "rainfall",
            "relative_humidity_2m_max":     "humidity_max",
            "relative_humidity_2m_min":     "humidity_min",
            "soil_moisture_0_to_7cm_mean":  "soil_moisture_raw",
            "et0_fao_evapotranspiration":   "evapotranspiration",
            "wind_speed_10m_max":           "wind_speed",
        })
        df["date"]         = pd.to_datetime(df["date"])
        df["temperature"]  = ((df["temp_max"] + df["temp_min"]) / 2).round(2)
        df["humidity"]     = ((df["humidity_max"] + df["humidity_min"]) / 2).round(2)
        df["soil_moisture"] = (df["soil_moisture_raw"] * 100).clip(0, 60).round(2)
        df = df.dropna(subset=["temperature", "humidity", "rainfall", "soil_moisture"])

        df["risk"] = label_risk(df)

        return df.reset_index(drop=True)

    except Exception as e:
        st.error(f"❌ Erreur API Open-Meteo : {e} — données simulées utilisées.")
        return _generate_fallback_data()


def _generate_fallback_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 5000
    dates    = pd.date_range("2010-01-01", periods=n, freq="D")
    temp     = np.random.uniform(18, 44, n)
    hum      = np.random.uniform(15, 95, n)
    rain     = np.random.uniform(0, 20, n)
    soil     = np.random.uniform(5, 55, n)
    wind     = np.random.uniform(5, 35, n)
    evap     = np.random.uniform(2, 8, n)
    risk = label_risk(pd.DataFrame({
        "temp_max": temp + 3, "temp_min": temp - 3,
        "temperature": temp, "humidity": hum,
        "humidity_max": hum + 8, "humidity_min": hum - 8,
        "rainfall": rain, "soil_moisture": soil,
        "evapotranspiration": evap, "wind_speed": wind,
    })).values
    return pd.DataFrame({
        "date": dates,
        "temperature": temp.round(2), "temp_max": (temp+3).round(2), "temp_min": (temp-3).round(2),
        "humidity": hum.round(2), "humidity_max": (hum+8).round(2), "humidity_min": (hum-8).round(2),
        "rainfall": rain.round(2), "soil_moisture": soil.round(2),
        "evapotranspiration": evap.round(2), "wind_speed": wind.round(1), "risk": risk,
    })

# ─────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────
FEATURES = [
    "temperature", "humidity", "rainfall", "soil_moisture",
    "temp_humidity_ratio", "hydric_stress_index", "log_rainfall",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
    df["hydric_stress_index"]  = (df["temperature"] - 25) / (df["soil_moisture"] + 1)
    df["log_rainfall"]         = np.log1p(df["rainfall"])
    return df

def feature_engineer_obs(obs: dict) -> dict:
    obs = obs.copy()
    obs["temp_humidity_ratio"] = obs["temperature"] / (obs["humidity"] + 1)
    obs["hydric_stress_index"]  = (obs["temperature"] - 25) / (obs["soil_moisture"] + 1)
    obs["log_rainfall"]         = np.log1p(obs["rainfall"])
    return obs

# ─────────────────────────────────────────────────────────
# Entraînement du modèle
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame):
    df_fe = add_features(_df)
    X = df_fe[FEATURES]
    y = df_fe["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    return model, scaler, X_test, y_test, X_te, auc

# ─────────────────────────────────────────────────────────
# Alertes
# ─────────────────────────────────────────────────────────
def generate_alerts(obs: dict) -> list[tuple[str, str, str]]:
    alerts = []
    if obs["temperature"] > 38:
        alerts.append(("critical", "🔴 Température critique",
                        f"{obs['temperature']:.1f}°C — stress thermique sévère, irrigation urgente."))
    elif obs["temperature"] > 33:
        alerts.append(("warning", "🟠 Température élevée",
                        f"{obs['temperature']:.1f}°C — surveiller l'évapotranspiration."))
    if obs["soil_moisture"] < 12:
        alerts.append(("critical", "🔴 Humidité sol critique",
                        f"{obs['soil_moisture']:.1f}% — en dessous du seuil vital (12%)."))
    elif obs["soil_moisture"] < 22:
        alerts.append(("warning", "🟠 Humidité sol faible",
                        f"{obs['soil_moisture']:.1f}% — irrigation préventive recommandée."))
    if obs["rainfall"] < 2:
        alerts.append(("warning", "🟠 Pluviométrie insuffisante",
                        f"{obs['rainfall']:.1f} mm — apport artificiel conseillé."))
    if obs["humidity"] < 30:
        alerts.append(("info", "🔵 Humidité air basse",
                        f"{obs['humidity']:.1f}% — évaporation accélérée possible."))
    if not alerts:
        alerts.append(("ok", "🟢 Conditions normales",
                        "Tous les paramètres sont dans les normes agronomiques."))
    return alerts

# ─────────────────────────────────────────────────────────
# Helper style graphiques
# ─────────────────────────────────────────────────────────
STYLE = {
    "temp": "#D85A30", "hum": "#378ADD", "soil": "#1D9E75",
    "rain": "#5DCAA5", "wind": "#BA7517", "risk": "#e53e3e",
    "bg": "#fafafa",
}

def style_ax(ax):
    ax.set_facecolor(STYLE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e0e0e0")
    ax.tick_params(colors="#888", labelsize=8)
    ax.yaxis.label.set_color("#555")
    ax.yaxis.label.set_size(9)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 BayAI")

    st.markdown("<div class='sidebar-sep'>Région agricole</div>", unsafe_allow_html=True)
    city = st.selectbox("", list(CITIES.keys()), label_visibility="collapsed")
    lat, lon = CITIES[city]

    st.markdown("<div class='sidebar-sep'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "", ["📊 Dashboard", "🤖 Prédiction", "📈 Modèle", "📋 Données"],
        label_visibility="collapsed",
    )

    # ── Filtre de dates ──────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='sidebar-sep'>Période d'analyse</div>", unsafe_allow_html=True)

    ABS_MIN = date(2010, 1, 1)
    ABS_MAX = date.today() - timedelta(days=date.today().weekday() + 7)

    # Valeurs par défaut stockées dans session_state (sans key sur le widget)
    if "fs" not in st.session_state:
        st.session_state["fs"] = ABS_MIN
    if "fe" not in st.session_state:
        st.session_state["fe"] = ABS_MAX

    # Boutons présélection : écrivent dans fs/fe puis st.rerun()
    # Placés AVANT le date_input pour que la value soit correcte au rendu
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("1 an",  use_container_width=True):
        try:
            st.session_state["fs"] = ABS_MAX.replace(year=ABS_MAX.year - 1)
        except ValueError:
            st.session_state["fs"] = ABS_MAX - timedelta(days=365)
        st.session_state["fe"] = ABS_MAX
        st.rerun()
    if col_p2.button("5 ans", use_container_width=True):
        try:
            st.session_state["fs"] = ABS_MAX.replace(year=ABS_MAX.year - 5)
        except ValueError:
            st.session_state["fs"] = ABS_MAX - timedelta(days=5 * 365)
        st.session_state["fe"] = ABS_MAX
        st.rerun()
    if col_p3.button("Tout",  use_container_width=True):
        st.session_state["fs"] = ABS_MIN
        st.session_state["fe"] = ABS_MAX
        st.rerun()

    # date_input SANS key → Streamlit ne bloque pas la mise à jour de value
    # La value est relue depuis fs/fe à chaque rerun
    date_range = st.date_input(
        "Période",
        value=(st.session_state["fs"], st.session_state["fe"]),
        min_value=ABS_MIN,
        max_value=ABS_MAX,
        format="DD/MM/YYYY",
        label_visibility="collapsed",
    )

    # Lecture du résultat : tuple de 2 quand les deux dates sont sélectionnées
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filter_start = max(date_range[0], ABS_MIN)
        filter_end   = min(date_range[1], ABS_MAX)
        # Persister pour les boutons suivants
        st.session_state["fs"] = filter_start
        st.session_state["fe"] = filter_end
    else:
        # En cours de saisie (1 seule date cliquée) → garder l'ancienne valeur
        filter_start = st.session_state["fs"]
        filter_end   = st.session_state["fe"]

    # Sécurité inversion
    if filter_end < filter_start:
        filter_start, filter_end = ABS_MIN, ABS_MAX
        st.session_state["fs"], st.session_state["fe"] = filter_start, filter_end

    st.caption(
        f"{filter_start.strftime('%d %b %Y')} → {filter_end.strftime('%d %b %Y')} "
        f"· {(filter_end - filter_start).days:,} jours"
    )

    # ── Chargement données complètes (cache) ─────────────
    with st.spinner("Chargement données…"):
        df_full = fetch_real_weather(lat, lon)

    # ── Vue filtrée utilisée dans tout le dashboard ──────
    df = df_full[
        (df_full["date"] >= pd.Timestamp(filter_start)) &
        (df_full["date"] <= pd.Timestamp(filter_end))
    ].reset_index(drop=True)

    # ── Modèle entraîné sur données complètes ────────────
    with st.spinner("Entraînement modèle…"):
        model, scaler, X_test, y_test, X_te, auc_score = train_model(df_full)

    st.markdown("---")
    st.markdown("<div class='sidebar-sep'>Dataset filtré</div>", unsafe_allow_html=True)
    st.metric("Observations", f"{len(df):,}")
    st.metric("Taux de risque", f"{df['risk'].mean():.1%}")

    st.markdown("<div class='sidebar-sep'>Modèle</div>", unsafe_allow_html=True)
    st.metric("Algorithme", "Random Forest")
    st.metric("ROC-AUC", f"{auc_score:.3f}")

    st.markdown(
        "<br><span class='sidebar-pill'>✅ Open-Meteo · sans clé API</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bayai-header">
  <div class="bayai-logo">🌾</div>
  <div>
    <div class="bayai-title">BayAI</div>
    <div class="bayai-sub">
      Détection du risque de sécheresse &nbsp;·&nbsp;
      {filter_start.strftime('%d %b %Y')} → {filter_end.strftime('%d %b %Y')} &nbsp;·&nbsp; {city}
    </div>
  </div>
</div>
<hr class="header-divider">
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # ── KPIs ──────────────────────────────────────────────
    last      = df.iloc[-1]
    risk_days = int(df["risk"].sum())
    risk_pct  = df["risk"].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "#D85A30", "Température moy.",
         f"{df['temperature'].mean():.1f}°C",
         f"max {df['temp_max'].mean():.1f} / min {df['temp_min'].mean():.1f}"),
        (k2, "#378ADD", "Humidité air",
         f"{df['humidity'].mean():.1f}%",
         f"max {df['humidity_max'].mean():.0f} / min {df['humidity_min'].mean():.0f}"),
        (k3, "#1D9E75", "Humidité sol",
         f"{df['soil_moisture'].mean():.1f}%",
         f"évapotransp. {df['evapotranspiration'].mean():.1f} mm/j"),
        (k4, "#5DCAA5", "Pluie moy./jour",
         f"{df['rainfall'].mean():.1f} mm",
         f"cumul total {df['rainfall'].sum():.0f} mm"),
        (k5, "#e53e3e", "Jours à risque",
         f"{risk_days:,}",
         f"{risk_pct:.1%} de la période"),
        (k6, "#BA7517", "Vent max moy.",
         f"{df['wind_speed'].mean():.1f} km/h",
         f"dernière donnée {last['date'].strftime('%d %b %Y')}"),
    ]
    for col, accent, label, value, sub in kpis:
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Série temporelle ───────────────────────────────────
    st.markdown("<div class='section-title'>Évolution temporelle 2010 → aujourd'hui</div>", unsafe_allow_html=True)

    fig_ts, axes_ts = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    fig_ts.patch.set_facecolor(STYLE["bg"])

    ts_vars = [
        ("temperature",  "Temp. moy. (°C)",   STYLE["temp"]),
        ("humidity",     "Humidité air (%)",   STYLE["hum"]),
        ("soil_moisture","Humidité sol (%)",   STYLE["soil"]),
        ("wind_speed",   "Vent max (km/h)",    STYLE["wind"]),
    ]
    for ax, (col, label, color) in zip(axes_ts, ts_vars):
        ax.plot(df["date"], df[col], color=color, linewidth=0.55, alpha=0.9)
        ax.fill_between(df["date"], df[col],
                        where=df["risk"] == 1,
                        color=STYLE["risk"], alpha=0.18, label="Zones à risque")
        ax.set_ylabel(label, fontsize=9)
        style_ax(ax)

    axes_ts[0].legend(fontsize=8, framealpha=0, loc="upper right")
    axes_ts[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes_ts[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0, fontsize=9)
    fig_ts.tight_layout(h_pad=0.4)
    st.pyplot(fig_ts)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 2 : distributions + saisonnalité ─────────────
    col_l, col_r = st.columns([1.15, 0.85], gap="large")

    with col_l:
        st.markdown("<div class='section-title'>Distribution par classe de risque</div>", unsafe_allow_html=True)
        fig_dist, axes_d = plt.subplots(2, 2, figsize=(9, 5.5))
        fig_dist.patch.set_facecolor(STYLE["bg"])
        dist_vars = [
            ("temperature",  "Température (°C)",    STYLE["temp"]),
            ("humidity",     "Humidité (%)",         STYLE["hum"]),
            ("rainfall",     "Précipitations (mm)",  STYLE["rain"]),
            ("soil_moisture","Humidité sol (%)",     STYLE["soil"]),
        ]
        for ax, (feat, label, color) in zip(axes_d.flat, dist_vars):
            for rv, alpha, lbl in [(0, 0.4, "Sans risque"), (1, 0.75, "Risque")]:
                subset = df[df["risk"] == rv][feat].dropna()
                ax.hist(subset, bins=30, alpha=alpha,
                        color=color if rv == 0 else STYLE["risk"],
                        label=lbl, density=True)
            ax.set_title(label, fontsize=10, fontweight="600", color="#2a3a2a", pad=4)
            ax.legend(fontsize=7.5, framealpha=0)
            style_ax(ax)
        fig_dist.tight_layout(pad=1.2)
        st.pyplot(fig_dist)
        plt.close()

    with col_r:
        st.markdown("<div class='section-title'>Saisonnalité du risque (moy. par mois)</div>", unsafe_allow_html=True)
        df_m = df.copy()
        df_m["month"] = df_m["date"].dt.month
        seasonal = df_m.groupby("month")["risk"].mean()
        mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

        fig_sea, ax_sea = plt.subplots(figsize=(6.5, 3.2))
        fig_sea.patch.set_facecolor(STYLE["bg"])
        bar_colors = [
            "#e53e3e" if v > 0.4 else "#f59e0b" if v > 0.2 else "#1D9E75"
            for v in seasonal.values
        ]
        ax_sea.bar(range(1, 13), seasonal.values, color=bar_colors, width=0.65, zorder=3)
        ax_sea.set_xticks(range(1, 13))
        ax_sea.set_xticklabels(mois, fontsize=8.5)
        ax_sea.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_sea.axhline(0.4, color="#e53e3e", linewidth=1.2,
                       linestyle="--", alpha=0.5, label="Seuil critique 40%")
        ax_sea.set_ylabel("% jours à risque", fontsize=9)
        ax_sea.legend(fontsize=8, framealpha=0)
        ax_sea.set_ylim(0, min(1.0, seasonal.max() * 1.35))
        style_ax(ax_sea)
        fig_sea.tight_layout()
        st.pyplot(fig_sea)
        plt.close()

        st.markdown("<div class='section-title' style='margin-top:16px'>Corrélation avec la cible risque</div>", unsafe_allow_html=True)
        corr_cols = ["temperature", "humidity", "rainfall", "soil_moisture", "wind_speed", "evapotranspiration"]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_vals = df[corr_cols + ["risk"]].corr()["risk"].drop("risk").sort_values()

        fig_corr, ax_corr = plt.subplots(figsize=(6.5, 3.0))
        fig_corr.patch.set_facecolor(STYLE["bg"])
        ax_corr.barh(corr_vals.index, corr_vals.values,
                     color=[STYLE["temp"] if v > 0 else STYLE["hum"] for v in corr_vals.values],
                     height=0.55)
        ax_corr.axvline(0, color="#aaa", linewidth=0.8)
        ax_corr.set_xlabel("Corrélation de Pearson", fontsize=9)
        style_ax(ax_corr)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close()


# ══════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ══════════════════════════════════════════════════════════
elif page == "🤖 Prédiction":
    st.markdown("<div class='section-title'>Simulateur de risque sécheresse</div>", unsafe_allow_html=True)
    st.caption("Ajustez les paramètres capteurs pour obtenir une prédiction en temps réel.")

    # Clés internes (_val_*) séparées des clés widget (sl_*) — évite le conflit Streamlit
    DEFAULTS = {"_val_temp": 27.5, "_val_hum": 68.9, "_val_rain": 5.0, "_val_soil": 36.0}
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:
        # Les scénarios modifient les clés internes AVANT que les sliders soient rendus
        st.markdown("<div class='section-title'>Scénarios rapides</div>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        if sc1.button("☀️ Sécheresse"):
            st.session_state["_val_temp"] = 41.0
            st.session_state["_val_hum"]  = 22.0
            st.session_state["_val_rain"] = 0.5
            st.session_state["_val_soil"] = 9.0
            st.rerun()
        if sc2.button("🌤️ Modéré"):
            st.session_state["_val_temp"] = 34.0
            st.session_state["_val_hum"]  = 42.0
            st.session_state["_val_rain"] = 3.0
            st.session_state["_val_soil"] = 19.0
            st.rerun()
        if sc3.button("🌿 Normal"):
            st.session_state["_val_temp"] = 23.0
            st.session_state["_val_hum"]  = 75.0
            st.session_state["_val_rain"] = 12.0
            st.session_state["_val_soil"] = 38.0
            st.rerun()

        st.markdown("<div class='section-title' style='margin-top:14px'>Paramètres capteurs</div>", unsafe_allow_html=True)
        # Clés widget différentes des clés internes → aucun conflit
        temp = st.slider("🌡️ Température (°C)",      10.0, 50.0,  st.session_state["_val_temp"], 0.5, key="sl_temp")
        hum  = st.slider("💨 Humidité air (%)",       10.0, 100.0, st.session_state["_val_hum"],  1.0, key="sl_hum")
        rain = st.slider("🌧️ Précipitations (mm/j)", 0.0,  30.0,  st.session_state["_val_rain"], 0.5, key="sl_rain")
        soil = st.slider("🌱 Humidité sol (%)",       0.0,  60.0,  st.session_state["_val_soil"], 0.5, key="sl_soil")

        # Synchroniser les valeurs internes avec les sliders (pour les conserver entre pages)
        st.session_state["_val_temp"] = temp
        st.session_state["_val_hum"]  = hum
        st.session_state["_val_rain"] = rain
        st.session_state["_val_soil"] = soil

        st.markdown("<div class='section-title' style='margin-top:16px'>Dernières valeurs réelles</div>", unsafe_allow_html=True)
        last = df.iloc[-1]
        st.caption(f"Mesure du {last['date'].strftime('%d %b %Y')}")
        lc1, lc2 = st.columns(2)
        lc1.metric("Température", f"{last['temperature']:.1f}°C",
                   delta=f"max {last['temp_max']:.1f}°C")
        lc2.metric("Humidité sol", f"{last['soil_moisture']:.1f}%")
        lc1.metric("Précipitations", f"{last['rainfall']:.1f} mm")
        lc2.metric("Vent max", f"{last['wind_speed']:.1f} km/h")

    with col_result:
        obs = {
            "temperature":   temp,
            "humidity":      hum,
            "rainfall":      rain,
            "soil_moisture": soil,
        }
        obs_fe   = feature_engineer_obs(obs)
        X_obs    = np.array([[obs_fe[f] for f in FEATURES]])
        X_obs_sc = scaler.transform(X_obs)
        proba    = float(model.predict_proba(X_obs_sc)[0, 1])
        pred     = int(proba >= 0.5)

        st.markdown("<div class='section-title'>Résultat ML</div>", unsafe_allow_html=True)
        if pred == 1:
            st.error(f"⚠️ **RISQUE DE SÉCHERESSE DÉTECTÉ** — {proba:.1%}")
        else:
            st.success(f"✅ **PAS DE RISQUE DÉTECTÉ** — {proba:.1%}")

        fig_g, ax_g = plt.subplots(figsize=(5, 0.85))
        fig_g.patch.set_facecolor(STYLE["bg"])
        gradient = np.linspace(0, 1, 300).reshape(1, -1)
        ax_g.imshow(gradient, aspect="auto", cmap="RdYlGn_r", extent=[0, 1, 0, 1])
        ax_g.axvline(proba, color="#111", linewidth=2.5)
        ax_g.set_yticks([])
        ax_g.set_xticks([0, 0.3, 0.6, 1.0])
        ax_g.set_xticklabels(["0%", "30%", "60%", "100%"], fontsize=9)
        ax_g.set_title(f"Score de risque : {proba:.1%}", fontsize=10, fontweight="600", pad=6)
        ax_g.spines[:].set_visible(False)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-title' style='margin-top:14px'>Importance des features</div>", unsafe_allow_html=True)
        fi_vals = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig_fi2, ax_fi2 = plt.subplots(figsize=(5, 2.8))
        fig_fi2.patch.set_facecolor(STYLE["bg"])
        ax_fi2.barh(fi_vals.index, fi_vals.values,
                    color=[STYLE["soil"] if v < 0.15 else STYLE["temp"] for v in fi_vals.values],
                    height=0.55)
        ax_fi2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        style_ax(ax_fi2)
        fig_fi2.tight_layout()
        st.pyplot(fig_fi2)
        plt.close()

        st.markdown("<div class='section-title' style='margin-top:14px'>Alertes agronomiques</div>", unsafe_allow_html=True)
        for level, title, msg in generate_alerts(obs):
            st.markdown(
                f"<div class='alert-box alert-{level}'><strong>{title}</strong><br>{msg}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-title' style='margin-top:14px'>Recommandation</div>", unsafe_allow_html=True)
        if proba >= 0.6:
            st.markdown("🚿 **Irriguer immédiatement** — activer le système d'arrosage, vérifier les tuyaux et surveiller l'humidité toutes les 6h.")
        elif proba >= 0.3:
            st.markdown("👀 **Surveillance renforcée** — planifier une irrigation préventive dans les 48h et vérifier les prévisions météo.")
        else:
            st.markdown("✔️ **Conditions favorables** — continuer le suivi standard. Prochaine vérification dans 24h.")


# ══════════════════════════════════════════════════════════
# PAGE 3 — MODÈLE
# ══════════════════════════════════════════════════════════
elif page == "📈 Modèle":
    st.markdown("<div class='section-title'>Performance du modèle Random Forest</div>", unsafe_allow_html=True)

    y_pred_prob = model.predict_proba(X_te)[:, 1]
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    for col, accent, label, value in [
        (m1, "#1D9E75", "Accuracy",  f"{accuracy_score(y_test, y_pred):.3f}"),
        (m2, "#378ADD", "Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}"),
        (m3, "#D85A30", "Recall",    f"{recall_score(y_test, y_pred, zero_division=0):.3f}"),
        (m4, "#BA7517", "ROC-AUC",   f"{auc_score:.3f}"),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='section-title'>Matrice de confusion</div>", unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor(STYLE["bg"])
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Pas risque", "Risque"],
            cmap="Greens", ax=ax_cm,
        )
        ax_cm.set_facecolor(STYLE["bg"])
        st.pyplot(fig_cm)
        plt.close()

        st.markdown("<div class='section-title' style='margin-top:16px'>Importance des features</div>", unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig_fi, ax_fi = plt.subplots(figsize=(5, 3.5))
        fig_fi.patch.set_facecolor(STYLE["bg"])
        ax_fi.barh(fi.index, fi.values,
                   color=[STYLE["soil"] if v < 0.15 else STYLE["temp"] for v in fi.values],
                   height=0.6)
        ax_fi.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        style_ax(ax_fi)
        fig_fi.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    with col_b:
        st.markdown("<div class='section-title'>Courbe ROC</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fig_roc.patch.set_facecolor(STYLE["bg"])
        ax_roc.plot(fpr, tpr, STYLE["hum"], linewidth=2.5, label=f"AUC = {auc_score:.3f}")
        ax_roc.fill_between(fpr, tpr, alpha=0.10, color=STYLE["hum"])
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_xlabel("Taux faux positifs", fontsize=9)
        ax_roc.set_ylabel("Taux vrais positifs", fontsize=9)
        ax_roc.legend(fontsize=9, framealpha=0)
        style_ax(ax_roc)
        fig_roc.tight_layout()
        st.pyplot(fig_roc)
        plt.close()

        st.markdown("<div class='section-title' style='margin-top:16px'>Rapport de classification</div>", unsafe_allow_html=True)
        report = classification_report(
            y_test, y_pred,
            target_names=["Pas de risque", "Risque"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).T.round(3)
            .style.background_gradient(cmap="Greens", subset=["f1-score"]),
            use_container_width=True,
        )

        st.markdown("<div class='section-title' style='margin-top:16px'>Seuil de décision</div>", unsafe_allow_html=True)
        threshold = st.slider("Ajuster le seuil de classification", 0.1, 0.9, 0.5, 0.05)
        y_pred_t  = (y_pred_prob >= threshold).astype(int)
        t1, t2, t3 = st.columns(3)
        t1.metric("Precision", f"{precision_score(y_test, y_pred_t, zero_division=0):.3f}")
        t2.metric("Recall",    f"{recall_score(y_test, y_pred_t, zero_division=0):.3f}")
        t3.metric("F1-Score",  f"{f1_score(y_test, y_pred_t, zero_division=0):.3f}")


# ══════════════════════════════════════════════════════════
# PAGE 4 — DONNÉES
# ══════════════════════════════════════════════════════════
elif page == "📋 Données":
    st.markdown("<div class='section-title'>Exploration du dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<span class='sidebar-pill'>✅ Open-Meteo · sans clé API · 2010 → semaine -1</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.selectbox("Filtrer par risque", ["Tous", "Risque = 0", "Risque = 1"])
    with col_f2:
        temp_range = st.slider(
            "Plage température (°C)",
            float(df["temperature"].min()), float(df["temperature"].max()),
            (float(df["temperature"].min()), float(df["temperature"].max())),
        )
    with col_f3:
        _max_rows     = max(10, len(df))
        _default_rows = min(50, _max_rows)
        n_rows = st.number_input("Lignes à afficher", 10, _max_rows, _default_rows, 10)

    df_filtered = df.copy()
    if risk_filter == "Risque = 0":
        df_filtered = df_filtered[df_filtered["risk"] == 0]
    elif risk_filter == "Risque = 1":
        df_filtered = df_filtered[df_filtered["risk"] == 1]
    df_filtered = df_filtered[
        df_filtered["temperature"].between(*temp_range)
    ].head(int(n_rows))

    display_cols = [
        "date", "temperature", "temp_max", "temp_min",
        "humidity", "humidity_max", "humidity_min",
        "rainfall", "soil_moisture", "evapotranspiration",
        "wind_speed", "risk",
    ]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    st.dataframe(
        df_filtered[display_cols].style.map(
            lambda v: "background-color:#fde8e8" if v == 1 else "background-color:#d1fae5",
            subset=["risk"],
        ),
        use_container_width=True,
        height=400,
    )
    st.markdown(f"**{len(df_filtered):,} lignes affichées sur {len(df):,} total**")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        city_slug   = city.split(",")[0].encode("ascii", "ignore").decode().strip().replace(" ", "_")
        start_slug  = filter_start.strftime("%Y%m%d")
        end_slug    = filter_end.strftime("%Y%m%d")
        st.download_button(
            "⬇️ Télécharger le dataset (CSV)",
            data=df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"bayai_{city_slug}_{start_slug}_{end_slug}.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.markdown("<div class='section-title'>Statistiques descriptives</div>", unsafe_allow_html=True)
        stat_cols = [c for c in ["temperature", "humidity", "rainfall", "soil_moisture", "wind_speed", "evapotranspiration"] if c in df_filtered.columns]
        st.dataframe(
            df_filtered[stat_cols].describe().round(2),
            use_container_width=True,
        )