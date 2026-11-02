import io
import math
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh

# ============================================================
# Hispanic Heritage Historical Data Explorer 2.0
# Single-file classroom app. Historical source: World Bank API.
# ============================================================

st.set_page_config(page_title="Hispanic Heritage Data & Function Explorer",
                   page_icon="📈", layout="wide")

st.title("Hispanic Heritage Historical Data & Function Explorer")
st.caption("Explore real historical data, fit mathematical models, and investigate how functions change.")

COUNTRIES = {
    "Mexico": "MEX", "United States": "USA", "Cuba": "CUB",
    "Puerto Rico": "PRI", "Dominican Republic": "DOM",
    "Guatemala": "GTM", "Honduras": "HND", "El Salvador": "SLV",
    "Nicaragua": "NIC", "Costa Rica": "CRI", "Panama": "PAN",
    "Colombia": "COL", "Venezuela": "VEN", "Ecuador": "ECU",
    "Peru": "PER", "Bolivia": "BOL", "Chile": "CHL",
    "Argentina": "ARG", "Uruguay": "URY", "Paraguay": "PRY",
    "Spain": "ESP", "Brazil": "BRA"
}
INDICATORS = {
    "Population, total": ("SP.POP.TOTL", "people"),
    "GDP per capita (current US$)": ("NY.GDP.PCAP.CD", "current US dollars per person"),
    "Life expectancy at birth, total (years)": ("SP.DYN.LE00.IN", "years"),
    "CO₂ emissions (metric tons per capita)": ("EN.ATM.CO2E.PC", "metric tons per person"),
    "School enrollment, primary (% gross)": ("SE.PRM.ENRR", "percent"),
    "Mortality rate, under-5 (per 1,000 live births)": ("SH.DYN.MORT", "deaths per 1,000 live births"),
    "Renewable electricity output (% of total)": ("EG.ELC.RNEW.ZS", "percent"),
}
CONTEXTS = {
    "Business": ("Revenue", "dollars", "time", "years", "Examine how a business measure changes over time."),
    "Engineering": ("Bridge load", "kilonewtons", "time", "seconds", "Model a measured engineering quantity over time."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "A simplified concentration model; not medical advice."),
    "Sports science": ("Position", "meters", "time", "seconds", "Study motion, speed, and acceleration."),
    "Biology": ("Population", "organisms", "time", "days", "Explore how a population changes."),
    "Bioengineering": ("Cell growth", "cells", "time", "hours", "Explore a simplified cell-growth model."),
    "Economic forecasting": ("Economic indicator", "index units", "time", "years", "Explore a trend; extrapolation is uncertain."),
    "Weather": ("Temperature", "°C", "time", "hours", "Explore a temperature pattern."),
    "Custom": ("Output", "units", "input", "units", "Describe your own mathematical context."),
}
SUGGESTED = {
    "Business": ["a*x+b", "a*x**2+b*x+c", "a*exp(k*x)"],
    "Engineering": ["a*x+b", "a*x**2+b*x+c", "a*sin(k*x)"],
    "Medical": ["a*exp(-k*x)", "a/(x+b)", "a*x+b"],
    "Sports science": ["a*x+b", "a*x**2+b*x+c", "a*sin(k*x)"],
    "Biology": ["a*exp(k*x)", "a/(1+exp(-k*(x-b)))", "a*x+b"],
    "Bioengineering": ["a*exp(k*x)", "a*x+b", "a*sin(k*x)"],
    "Economic forecasting": ["a*x+b", "a*x**2+b*x+c", "a*log(x)+b"],
    "Weather": ["a*sin(k*x)+b", "a*x+b", "a*cos(k*x)+b"],
    "Custom": ["a*x+b", "a*x**2+b*x+c", "a*exp(k*x)"],
}

def wb_fetch(country_code, indicator_code, start_year, end_year):
    url = (f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
           f"{indicator_code}?date={start_year}:{end_year}&format=json&per_page=2000")
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"])
    rows = [{"year": int(r["date"]), "value": float(r["value"])}
            for r in payload[1] if r.get("value") is not None]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country_code, indicator_code, start_year, end_year):
    return wb_fetch(country_code, indicator_code, start_year, end_year)

def parse_function(source, variable):
    """Parse common classroom notation safely with SymPy."""
    source = source.strip().replace("^", "**")
    source = source.replace("ln(", "log(")
    if len(source) > 250:
        raise ValueError("Function is too long.")
    allowed = {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs"}
    locals_map = {variable: sp.Symbol(variable, real=True), "pi": sp.pi, "E": sp.E}
    locals_map.update({name: getattr(sp, name) for name in allowed})
    try:
        expr = sp.sympify(source, locals=locals_map, evaluate=True)
    except Exception as exc:
        raise ValueError("I could not read that expression. Use notation such as x**2, exp(x), log(x), or sin(x).") from exc
    symbols = expr.free_symbols
    if symbols - {locals_map[variable]}:
        raise ValueError("Use only the selected input variable and numeric constants.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The expression contains an invalid or undefined value.")
    return expr, locals_map[variable]

def numeric_value(expr, symbol, value):
    try:
        result = complex(sp.N(expr.subs(symbol, float(value))))
        if abs(result.imag) > 1e-8 or not np.isfinite(result.real):
            return np.nan
        return float(result.real)
    except Exception:
        return np.nan

def safe_eval_array(expr, symbol, xs):
    return np.array([numeric_value(expr, symbol, x) for x in xs], dtype=float)

def find_roots_numeric(expr, symbol, lo, hi, samples=1600):
    """Numerically locate real roots, with sign-change bracketing and near-zero checks."""
    grid = np.linspace(lo, hi, samples)
    vals = safe_eval_array(expr, symbol, grid)
    roots = []
    for i in range(len(grid)-1):
        a, b, fa, fb = grid[i], grid[i+1], vals[i], vals[i+1]
        if np.isfinite(fa) and abs(fa) < 1e-7:
            roots.append(a)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            try:
                roots.append(brentq(lambda z: numeric_value(expr, symbol, z), a, b))
            except Exception:
                pass
    if np.isfinite(vals[-1]) and abs(vals[-1]) < 1e-7:
        roots.append(grid[-1])
    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-4:
            unique.append(float(r))
    return unique

def derivative_analysis(expr, x, lo, hi):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    critical = find_roots_numeric(d1, x, lo, hi)
    inflect_candidates = find_roots_numeric(d2, x, lo, hi)
    eps = max((hi-lo)*1e-5, 1e-5)
    extrema, inflections = [], []
    for p in critical:
        left, right = numeric_value(d1, x, max(lo, p-eps)), numeric_value(d1, x, min(hi, p+eps))
        y = numeric_value(expr, x, p)
        if np.isfinite(left) and np.isfinite(right):
            if left > 0 and right < 0:
                kind = "local maximum"
            elif left < 0 and right > 0:
                kind = "local minimum"
            else:
                kind = "stationary point (not classified as an extremum)"
            extrema.append((p, y, kind))
    for p in inflect_candidates:
        left, right = numeric_value(d2, x, max(lo, p-eps)), numeric_value(d2, x, min(hi, p+eps))
        if np.isfinite(left) and np.isfinite(right) and left * right < 0:
            inflections.append((p, numeric_value(expr, x, p)))
    # Interval extrema include endpoints; numerical search avoids undefined values.
    grid = np.linspace(lo, hi, 4000)
    ys = safe_eval_array(expr, x, grid)
    finite = np.isfinite(ys)
    abs_extrema = None
    if finite.any():
        valid_y = ys[finite]
        valid_x = grid[finite]
        abs_extrema = (float(valid_x[np.argmin(valid_y)]), float(np.min(valid_y)),
                       float(valid_x[np.argmax(valid_y)]), float(np.max(valid_y)))
    return d1, d2, critical, extrema, inflections, abs_extrema

def fmt(v, digits=6):
    if v is None or not np.isfinite(v):
        return "undefined"
    return f"{v:.{digits}g}"

def make_excel(summary, data, results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(list(summary.items()), columns=["Field", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
    output.seek(0)
    return output.getvalue()

def make_pdf(summary, data, results, fig):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.45*inch, leftMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer — Report", styles["Title"])]
    story.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 10))
    rows = [["Metadata", "Value"]] + [[str(k), str(v)] for k, v in summary.items()]
    tab = Table(rows, colWidths=[1.65*inch, 5.4*inch])
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.35,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.extend([tab, Spacer(1, 10)])
    if not data.empty:
        fig_m, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(data["year"], data["value"], label="Observed data", s=14)
        ax.set_xlabel("Year"); ax.set_ylabel(summary.get("Units", "Value")); ax.grid(True, alpha=.25); ax.legend()
        chart_buf = io.BytesIO()
        fig_m.tight_layout(); fig_m.savefig(chart_buf, format="png", dpi=140); plt.close(fig_m); chart_buf.seek(0)
        story.append(Image(chart_buf, width=6.8*inch, height=2.55*inch))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    story.append(Paragraph("Polynomial fits describe the selected historical sample. In-sample fit metrics are not guarantees of future accuracy. Extrapolation can become implausible, and residuals/MAE describe typical historical fit error rather than a validated prediction interval.", styles["BodyText"]))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    preview = [["Year", "Observed value"]] + [[str(int(r.year)), fmt(r.value)] for r in data.head(12).itertuples()]
    t = Table(preview, colWidths=[1.2*inch, 2.0*inch])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
    story.append(t)
    story.append(Paragraph("Calculated year results", styles["Heading2"]))
    if not results.empty:
        rr = [["Year", "Estimate", "Classification"]] + [[str(r.year), fmt(r.estimate), r.classification] for r in results.itertuples()]
        rt = Table(rr, colWidths=[1.0*inch, 1.5*inch, 3.0*inch])
        rt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        story.append(rt)
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# Sidebar: source and data selection
with st.sidebar:
    st.header("1. Historical data")
    country_name = st.selectbox("Country / territory", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    start_year = st.number_input("First data year", min_value=1960, max_value=2025, value=1990, step=1)
    end_year = st.number_input("Last data year", min_value=int(start_year), max_value=2025, value=2023, step=1)
    st.caption("Source: World Bank Open Data API. Missing years are not filled in.")
    load = st.button("Load / refresh data", type="primary", use_container_width=True)
    st.divider()
    st.header("2. Course level")
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    st.header("3. Model")
    degree = st.slider("Polynomial degree", 1, 6, 2)
    holdout_pct = st.slider("Chronological holdout (%)", 0, 40, 20, 5)
    sample_step = st.slider("Model marker / sample-table increment", 1, 10, 3)
    st.header("4. Context")
    context = st.selectbox("Application context", list(CONTEXTS))
    custom_context = st.text_input("Context description", CONTEXTS[context][4])
    suggested = SUGGESTED[context]
    function_choice = st.selectbox("Function setup", ["Use polynomial fit", "Suggested function", "Build with parameters", "Enter my own function"])
    suggested_fn = st.selectbox("Suggested form", suggested)
    if function_choice == "Build with parameters":
        st.caption("Edit coefficients below. Use x as the input variable.")
        a = st.number_input("a", value=2.0)
        b = st.number_input("b", value=1.0)
        c = st.number_input("c", value=0.0)
        k = st.number_input("k", value=0.1, format="%.4f")
        parameter_fn = st.text_input("Parameter-based function", value="a*x+b")
        for parameter_name, parameter_value in [("a", a), ("b", b), ("c", c), ("k", k)]:
            parameter_fn = re.sub(rf"\\b{parameter_name}\\b", f"({parameter_value})", parameter_fn)
    else:
        parameter_fn = "x"
    custom_fn = st.text_input("Your function (use x)", value="x**2" if function_choice == "Enter my own function" else "2*x+1")
    st.header("5. Variable labels")
    x_label = st.text_input("Independent variable", CONTEXTS[context][2])
    x_units = st.text_input("Input units", CONTEXTS[context][3])
    y_label = st.text_input("Dependent variable", CONTEXTS[context][0])
    y_units = st.text_input("Output units", CONTEXTS[context][1])

# Data load
country_code = COUNTRIES[country_name]
indicator_code, indicator_units = INDICATORS[indicator_name]
data_key = (country_code, indicator_code, int(start_year), int(end_year))
if "loaded_key" not in st.session_state:
    st.session_state.loaded_key = None
if load or st.session_state.loaded_key != data_key:
    try:
        with st.spinner("Retrieving World Bank observations..."):
            df = cached_wb_fetch(country_code, indicator_code, int(start_year), int(end_year))
        st.session_state.raw_data = df
        st.session_state.loaded_key = data_key
        st.session_state.calculated = {}
    except Exception as e:
        st.error(f"Data request failed: {e}. Check the connection, API status, and selected indicator.")
        st.stop()
df = st.session_state.get("raw_data", pd.DataFrame(columns=["year","value"])).copy()
# Calculated outputs depend on the source selection and polynomial degree.
model_signature = (data_key, int(degree), len(df), tuple(df.get("year", pd.Series(dtype=int)).tolist()))
if st.session_state.get("model_signature") != model_signature:
    st.session_state.calculated = {}
    st.session_state.model_signature = model_signature
if df.empty:
    st.warning("No observations were returned for this selection. Choose another country, indicator, or year range.")
    st.stop()

df = df.dropna().drop_duplicates("year").sort_values("year").reset_index(drop=True)
actual_min, actual_max = int(df.year.min()), int(df.year.max())
mean_value = float(df.value.mean())
st.caption(f"Loaded {len(df)} observations for {country_name} · {indicator_name} · {actual_min}–{actual_max}. Units: {indicator_units}.")

# Fit polynomial with centered time; preserve exact transform t = year - base year.
base_year = actual_min
t = df.year.to_numpy(dtype=float) - base_year
y = df.value.to_numpy(dtype=float)
fit_error = None
poly_coeff = None
if len(np.unique(df.year)) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree+1} distinct observed years; only {len(np.unique(df.year))} are available.")
    st.stop()
try:
    poly_coeff = np.polyfit(t, y, degree)
    poly = np.poly1d(poly_coeff)
    fitted = poly(t)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - mean_value)**2))
    r2 = 1 - ss_res/ss_tot if ss_tot else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    df["fitted"] = fitted
    df["residual"] = residuals
except Exception as e:
    st.error(f"Could not fit the selected polynomial: {e}")
    st.stop()

# Chronological holdout: train on earliest observations, evaluate on latest.
holdout_score = None
if holdout_pct > 0 and len(df) >= max(degree + 4, 8):
    n_test = max(1, int(round(len(df)*holdout_pct/100)))
    n_train = len(df)-n_test
    if n_train >= degree+1:
        train_t = df.year.iloc[:n_train].to_numpy(dtype=float)-base_year
        test_t = df.year.iloc[n_train:].to_numpy(dtype=float)-base_year
        hp = np.poly1d(np.polyfit(train_t, df.value.iloc[:n_train], degree))
        pred = hp(test_t)
        holdout_score = float(np.mean(np.abs(df.value.iloc[n_train:].to_numpy()-pred)))

# Choose the active function
x = sp.Symbol("x", real=True)
active_expr = None
function_error = None
if function_choice == "Use polynomial fit":
    # Convert centered-time polynomial to an expression in the displayed year variable x.
    active_expr = sp.Float(poly_coeff[0]) * (x-base_year)**degree
    for i, coefficient in enumerate(poly_coeff[1:], start=1):
        active_expr += sp.Float(coefficient) * (x-base_year)**(degree-i)
elif function_choice == "Suggested function":
    try:
        active_expr, _ = parse_function(suggested_fn, "x")
    except ValueError as e:
        function_error = str(e)
elif function_choice == "Build with parameters":
    try:
        active_expr, _ = parse_function(parameter_fn, "x")
    except ValueError as e:
        function_error = str(e)
else:
    try:
        active_expr, _ = parse_function(custom_fn, "x")
    except ValueError as e:
        function_error = str(e)

# Fit metrics always describe the retrieved-data polynomial, not a different selected function.
st.subheader("Historical fit")
m1, m2, m3, m4 = st.columns(4)
m1.metric("In-sample R²", fmt(r2, 5))
m2.metric("In-sample MAE", fmt(mae))
m3.metric("Observations", len(df))
m4.metric("Historical mean", fmt(mean_value))
st.write(f"**Centered-time equation:** `t = year − {base_year}`")
coeff_text = " + ".join([f"({fmt(c, 10)})·t^{degree-i}" for i,c in enumerate(poly_coeff)])
st.code(f"fitted value = {coeff_text}\nBase year = {base_year}; degree = {degree}", language="text")
st.caption(f"Historical range: {actual_min}–{actual_max}. Residuals are observed − fitted values. Metrics above are in-sample.")
if holdout_score is not None:
    st.info(f"Chronological holdout MAE (separate score): {fmt(holdout_score)} using the latest {n_test} observations. This is not an in-sample metric.")
else:
    st.caption("Chronological holdout comparison unavailable: not enough observations for this degree and split.")
with st.expander("View residuals and fitted values at observed years"):
    st.dataframe(df[["year","value","fitted","residual"]].rename(columns={"year":"Year","value":"Actual observation","fitted":"Fitted value","residual":"Residual"}), use_container_width=True)

if function_error:
    st.error(function_error)
    st.stop()
if active_expr is None:
    st.error("No valid active function is available.")
    st.stop()

st.subheader("Function workspace")
st.latex(sp.latex(sp.Eq(sp.Function("f")(x), active_expr)))
st.caption(f"Context: {custom_context} | {y_label} is measured in {y_units}; {x_label} is measured in {x_units}.")
if function_choice != "Use polynomial fit":
    st.warning("The active function is a student-selected model, not the historical polynomial fit. Historical fit metrics still describe only the polynomial fitted to the retrieved observations.")

# Calculus controls and domain
left, right = st.columns(2)
with left:
    interval_lo = st.number_input("Analysis interval start", value=float(actual_min), step=1.0)
with right:
    interval_hi = st.number_input("Analysis interval end", value=float(actual_max), step=1.0)
if interval_hi <= interval_lo:
    st.error("The analysis interval end must be greater than its start.")
    st.stop()
lo, hi = float(interval_lo), float(interval_hi)
st.caption("The graph and numerical analysis use this selected interval. For rational/logarithmic functions, undefined regions are left as gaps.")

# Selected input point and secant interval
if "point_x" not in st.session_state:
    st.session_state.point_x = float(np.clip((lo+hi)/2, lo, hi))
st.session_state.point_x = float(np.clip(st.session_state.point_x, lo, hi))
point_x = st.slider("Current input value", min_value=float(lo), max_value=float(hi),
                    value=float(st.session_state.point_x), step=max((hi-lo)/500, 0.001))
st.session_state.point_x = point_x
s1, s2 = st.columns(2)
with s1:
    sec_a = st.slider("Average-rate interval start", float(lo), float(hi), float(lo), step=max((hi-lo)/500, .001))
with s2:
    sec_b = st.slider("Average-rate interval end", float(lo), float(hi), float(hi), step=max((hi-lo)/500, .001))
if sec_b <= sec_a:
    st.warning("Choose an interval end greater than its start to calculate average rate of change.")
fx = numeric_value(active_expr, x, point_x)
d1_expr, d2_expr = sp.diff(active_expr, x), sp.diff(active_expr, x, 2)
slope = numeric_value(d1_expr, x, point_x)
accel = numeric_value(d2_expr, x, point_x)
f_a, f_b = numeric_value(active_expr, x, sec_a), numeric_value(active_expr, x, sec_b)
avg_rate = (f_b-f_a)/(sec_b-sec_a) if sec_b > sec_a and np.isfinite(f_a) and np.isfinite(f_b) else np.nan

# Plot function with NaNs breaking curves at discontinuities.
grid = np.linspace(lo, hi, 1000)
grid_y = safe_eval_array(active_expr, x, grid)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.year, y=df.value, mode="markers", name="Observed historical data",
                         marker=dict(size=7)))
fit_grid = np.linspace(actual_min, actual_max, 500)
fit_y = poly(fit_grid-base_year)
fig.add_trace(go.Scatter(x=fit_grid, y=fit_y, mode="lines", name="Historical polynomial fit"))
fig.add_vline(x=actual_min, line_dash="dash", annotation_text="Historical boundary")
if hi > actual_max:
    fig.add_trace(go.Scatter(x=grid[grid>actual_max], y=grid_y[grid>actual_max], mode="lines", name="Outside historical range (model)"))
fig.add_trace(go.Scatter(x=grid, y=grid_y, mode="lines", name="Selected function"))
# Sample increment affects model markers.
sample_x = grid[::sample_step]
sample_y = safe_eval_array(active_expr, x, sample_x)
fig.add_trace(go.Scatter(x=sample_x, y=sample_y, mode="markers", name=f"Model samples (every {sample_step})", marker=dict(size=5)))
if np.isfinite(fx):
    fig.add_trace(go.Scatter(x=[point_x], y=[fx], mode="markers", name="Current point", marker=dict(size=12, symbol="diamond")))
    if np.isfinite(slope):
        tangent_x = np.array([point_x-(hi-lo)*.08, point_x+(hi-lo)*.08])
        tangent_y = fx+slope*(tangent_x-point_x)
        fig.add_trace(go.Scatter(x=tangent_x, y=tangent_y, mode="lines", name="Tangent line", line=dict(dash="dot")))
if sec_b > sec_a and np.isfinite(avg_rate):
    sec_x = np.array([sec_a, sec_b])
    sec_y = np.array([f_a, f_b])
    fig.add_trace(go.Scatter(x=sec_x, y=sec_y, mode="lines+markers", name="Secant line", line=dict(dash="dash")))
fig.update_layout(height=520, xaxis_title=f"{x_label} ({x_units})",
                  yaxis_title=f"{y_label} ({y_units})", legend=dict(orientation="h", y=-.22),
                  margin=dict(t=30,b=100))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Value and rates")
c1,c2,c3,c4 = st.columns(4)
c1.metric(f"f({fmt(point_x,5)})", fmt(fx))
c2.metric("Instantaneous rate f′(x)", fmt(slope))
c3.metric("Second derivative f″(x)", fmt(accel))
c4.metric("Average rate on interval", fmt(avg_rate))
st.write(f"- **Function value:** {y_label} is {fmt(fx)} {y_units} at {fmt(point_x)} {x_units}.")
st.write(f"- **First derivative:** the instantaneous change is {fmt(slope)} {y_units} per {x_units}.")
st.write(f"- **Second derivative:** the rate itself changes at {fmt(accel)} {y_units} per {x_units}².")
st.write(f"- **Average rate:** from {fmt(sec_a)} to {fmt(sec_b)}, the average change is {fmt(avg_rate)} {y_units} per {x_units}.")
if context == "Sports science":
    st.caption("If the dependent variable is position, the first derivative represents velocity and the second derivative represents acceleration.")
if context == "Business" or context == "Economic forecasting":
    st.caption("A positive first derivative indicates growth; a negative one indicates decline. The second derivative describes whether that growth/decline rate is increasing or decreasing.")

# Course-appropriate analysis
st.subheader(f"{course} analysis")
if course == "Honors Algebra 2":
    st.write(f"**Domain:** values where the expression is defined. **Range on the selected interval:** approximately {fmt(np.nanmin(grid_y))} to {fmt(np.nanmax(grid_y))}, when defined.")
    st.write("**Graphical trend:** use the curve and its increasing/decreasing sections to describe the pattern. **End behavior:** depends on the function type and leading terms; the displayed interval is not necessarily the full domain.")
    st.write(f"**Average rate of change:** {fmt(avg_rate)} {y_units} per {x_units} on the selected interval.")
elif course == "Precalculus":
    st.write(f"**Degree / coefficients:** for the historical polynomial, degree {degree} controls the possible number of turning points and the leading coefficient influences end behavior. For other function types, inspect their parameters and shape.")
    st.write(f"**Average rate of change:** {fmt(avg_rate)} {y_units} per {x_units}. Changing average rates suggest changing steepness and may indicate turning points or changing concavity.")
    st.write("Turning points and concavity depend on derivative sign changes; a derivative equal to zero alone does not prove a maximum or minimum.")
else:
    st.latex(r"f'(x)="+sp.latex(d1_expr))
    st.latex(r"f''(x)="+sp.latex(d2_expr))
    d1, d2, critical, extrema, inflections, abs_extrema = derivative_analysis(active_expr, x, lo, hi)
    st.write("**Critical numbers in the selected interval:** " + (", ".join(fmt(v) for v in critical) if critical else "none found numerically"))
    if extrema:
        for p, val, kind in extrema:
            st.write(f"- x ≈ {fmt(p)}, f(x) ≈ {fmt(val)}: {kind}, based on the sign change of f′.")
    else:
        st.write("No local extrema were classified by a first-derivative sign change in this interval.")
    st.write("**Inflection points (verified concavity change):** " + (", ".join(f"({fmt(p)}, {fmt(v)})" for p,v in inflections) if inflections else "none found"))
    if abs_extrema:
        st.write(f"**Numerical absolute extrema on the selected closed interval:** minimum ≈ ({fmt(abs_extrema[0])}, {fmt(abs_extrema[1])}); maximum ≈ ({fmt(abs_extrema[2])}, {fmt(abs_extrema[3])}). This numerical scan may miss narrow features or singularities.")
    st.write("**Increasing/decreasing and concavity:** infer intervals from the signs of f′ and f″ between critical numbers and verified inflection points. Numerical roots are approximations.")
    st.write("**Domain restrictions:** logarithms require positive arguments; denominators cannot equal zero; even roots require nonnegative radicands. The plotted curve is broken wherever evaluation is undefined.")

# Single-year calculation
st.subheader("Calculate a year")
single_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=int(np.clip(actual_max,1950,2100)), step=1)
if st.button("Calculate Value", type="primary"):
    st.session_state.calculated["single"] = int(single_year)
single = st.session_state.calculated.get("single")
if single is not None:
    estimate = float(poly(single-base_year))
    classification = "within historical fitted range" if actual_min <= single <= actual_max else "outside historical range (extrapolation)"
    st.write(f"**Year:** {single} · **Estimate:** {fmt(estimate)} {indicator_units} · **Fitted range:** {actual_min}–{actual_max} · **Classification:** {classification}")
    observed_row = df[df.year == single]
    if not observed_row.empty:
        st.caption(f"An actual observation exists for this year: {fmt(float(observed_row.value.iloc[0]))} {indicator_units}. The fitted estimate is separate from the observed value.")
    else:
        st.caption("No actual observation for this year; this is a model estimate, not an observed value.")
    if single < actual_min or single > actual_max:
        st.warning("This is extrapolation. Polynomial behavior outside the observed years may become implausible; the app does not clip the estimate.")
    if abs(estimate) > max(abs(y).max()*20, 1e9):
        st.warning("The estimate is unusually far from the observed scale. Treat it as potentially implausible.")

# Batch years
st.subheader("Batch year estimates")
batch_mode = st.radio("Batch input", ["Year range", "Comma-separated custom years"], horizontal=True)
batch_years = []
invalid_inputs = []
if batch_mode == "Year range":
    r1,r2,rstep = st.columns(3)
    with r1: batch_start = st.number_input("Range start", 1950, 2100, int(np.clip(actual_min,1950,2100)), key="brs")
    with r2: batch_end = st.number_input("Range end", 1950, 2100, int(np.clip(actual_max,1950,2100)), key="bre")
    with rstep: batch_step = st.number_input("Step (1–10)", 1, 10, 5)
    if batch_end < batch_start:
        invalid_inputs.append("Range end must be greater than or equal to range start.")
    else:
        batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
else:
    custom_years = st.text_input("Years separated by commas", value="2000, 2010, 2020, 2030")
    for item in custom_years.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            val = int(item)
            if 1950 <= val <= 2100:
                batch_years.append(val)
            else:
                invalid_inputs.append(f"{item}: year must be from 1950 to 2100.")
        except ValueError:
            invalid_inputs.append(f"{item}: not a whole-number year.")
batch_years = sorted(set(batch_years))
if invalid_inputs:
    st.warning("Invalid input(s): " + " ".join(invalid_inputs))
if batch_years:
    batch_results = []
    for yr in batch_years:
        est = float(poly(yr-base_year))
        observed = df[df.year == yr]
        batch_results.append({"year":yr, "estimate":est,
                              "classification":"within fitted range" if actual_min<=yr<=actual_max else "outside fitted range",
                              "actual_observation":float(observed.value.iloc[0]) if not observed.empty else np.nan})
    results_df = pd.DataFrame(batch_results)
    st.dataframe(results_df.rename(columns={"year":"Year","estimate":"Estimate","classification":"Range classification","actual_observation":"Actual observation"}), use_container_width=True)
    if single is not None and single in batch_years:
        row = results_df[results_df.year==single].iloc[0]
        st.caption(f"Consistency check: batch estimate for {single} is {fmt(row.estimate)}; single-year estimate is {fmt(poly(single-base_year))}.")
    if st.button("Save batch results for export"):
        st.session_state.calculated["batch"] = results_df.to_dict("records")
else:
    results_df = pd.DataFrame(columns=["year","estimate","classification","actual_observation"])
    st.caption("Enter at least one valid year to create batch results.")

# Visual simulation and animation controls
st.subheader("Interactive visual simulation")
visual = st.selectbox("Choose a visual", ["Moving object", "Walking person / prosthetic-leg application (conceptual)", "Tank / concentration indicator", "Population / cell colony", "Thermometer / weather", "Business indicator"])
play_col, pause_col, reset_col, speed_col = st.columns(4)
with play_col:
    if st.button("▶ Play"): st.session_state.playing = True
with pause_col:
    if st.button("Ⅱ Pause"): st.session_state.playing = False
with reset_col:
    if st.button("↺ Reset"):
        st.session_state.playing = False
        st.session_state.point_x = lo
with speed_col:
    speed = st.selectbox("Speed", ["Slow","Normal","Fast"], index=1)
if "playing" not in st.session_state: st.session_state.playing = False
if st.session_state.playing:
    interval_ms = {"Slow":1000,"Normal":500,"Fast":200}[speed]
    st_autorefresh(interval=interval_ms, key="animation_refresh")
    point_x = lo + ((point_x-lo)+(hi-lo)/100) % (hi-lo)
    st.session_state.point_x = point_x
    st.info(f"Playing simulation at x ≈ {fmt(point_x)}. Pause or reset at any time.")
# The same selected function value drives the numeric simulation indicator.
sim_value = numeric_value(active_expr, x, point_x)
if visual == "Walking person / prosthetic-leg application (conceptual)":
    st.info("Conceptual classroom application: a moving point represents a changing measured quantity during a gait cycle. This is not a clinical or prosthetic design tool.")
elif visual == "Tank / concentration indicator":
    st.info("Conceptual tank/concentration gauge. The model value is mapped to a display scale; it is not dosing guidance.")
elif visual == "Population / cell colony":
    st.info("Conceptual population/cell-growth indicator. The display scale is normalized to the selected interval.")
elif visual == "Thermometer / weather":
    st.info("Conceptual thermometer. Values outside the familiar temperature range remain visible as model outputs.")
elif visual == "Business indicator":
    st.info("Conceptual business gauge. Positive/negative rates indicate growth/decline in the modeled output.")
else:
    st.info("Conceptual moving-object indicator. If y is position, f′ is velocity and f″ is acceleration.")
if np.isfinite(sim_value):
    st.metric("Simulation output at moving point", f"{fmt(sim_value)} {y_units}")
else:
    st.warning("The selected function is undefined at the current simulation point.")

# Country comparison
st.subheader("Compare countries — same indicator")
compare_names = st.multiselect("Countries to compare", list(COUNTRIES), default=[country_name, "Mexico"] if country_name != "Mexico" else ["Mexico","Colombia"])
compare_fig = go.Figure()
comparison_frames = []
for name in compare_names:
    try:
        cdf = cached_wb_fetch(COUNTRIES[name], indicator_code, int(start_year), int(end_year))
        if not cdf.empty:
            compare_fig.add_trace(go.Scatter(x=cdf.year, y=cdf.value, mode="lines+markers", name=name))
            comparison_frames.append(cdf.assign(country=name))
    except Exception:
        st.warning(f"Could not retrieve comparison data for {name}.")
if comparison_frames:
    compare_fig.update_layout(height=380, xaxis_title="Year", yaxis_title=f"{indicator_name} ({indicator_units})")
    st.plotly_chart(compare_fig, use_container_width=True)
    st.caption("Countries may have different missing years and coverage. Lines connect only the available observations within each country's returned series; gaps and unequal coverage affect comparisons.")
else:
    st.caption("Choose at least one country to compare.")

# Printer-friendly report
st.subheader("Report and exports")
printer_mode = st.toggle("Printer mode (simplified report panel)", value=False)
summary = {
    "Country":country_name, "Indicator":indicator_name, "Indicator code":indicator_code,
    "Units":indicator_units, "Data source":"World Bank Open Data API",
    "Observed years":f"{actual_min}–{actual_max}", "Observation count":len(df),
    "Polynomial degree":degree, "Base year":base_year, "Centered time":"t = year - base year",
    "In-sample R-squared":r2, "In-sample MAE":mae, "Historical mean":mean_value,
    "Chronological holdout MAE":holdout_score if holdout_score is not None else "not available",
    "Active function":str(active_expr), "Function type":function_choice,
    "Context":context, "Course":course, "Edited data status":"No manual data edits in this version"
}
if printer_mode:
    st.markdown("### Classroom report")
    st.write(f"**{country_name} — {indicator_name}** ({actual_min}–{actual_max})")
    st.write(f"Degree-{degree} polynomial; base year {base_year}; in-sample R² {fmt(r2)}; in-sample MAE {fmt(mae)}.")
    st.write("Residuals and MAE describe fit error, not a validated 95% prediction interval. Extrapolation is uncertain and may be implausible.")
    st.dataframe(df[["year","value","fitted","residual"]].head(15), use_container_width=True)
else:
    st.write("Downloads reflect the current country, indicator, degree, active context/function, and calculated-year results. Data editing is not enabled; source observations remain unchanged.")
export_results = pd.DataFrame(st.session_state.calculated.get("batch", []))
if single is not None:
    single_est = float(poly(single-base_year))
    single_row = pd.DataFrame([{"year":single,"estimate":single_est,
                                "classification":"within fitted range" if actual_min<=single<=actual_max else "outside fitted range",
                                "actual_observation":float(df.loc[df.year==single,"value"].iloc[0]) if (df.year==single).any() else np.nan}])
    export_results = pd.concat([single_row, export_results], ignore_index=True).drop_duplicates("year", keep="first")
csv_bytes = df.to_csv(index=False).encode("utf-8")
xlsx_bytes = make_excel(summary, df, export_results)
pdf_bytes = make_pdf(summary, df, export_results, fig)
d1,d2,d3 = st.columns(3)
d1.download_button("Download CSV", csv_bytes, "historical_working_data.csv", "text/csv", use_container_width=True)
d2.download_button("Download Excel (.xlsx)", xlsx_bytes, "hispanic_heritage_report.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d3.download_button("Download PDF report", pdf_bytes, "hispanic_heritage_report.pdf",
                   "application/pdf", use_container_width=True)

with st.expander("How the app works, assumptions, and limitations"):
    st.markdown("""
    **Data flow:** country and indicator selections are sent to the World Bank API. Returned observations are sorted by year, missing values are omitted, and a polynomial is fit to the available observations. The time variable is centered as `t = year - base year` to improve numerical conditioning. The displayed-year function uses that same centered transformation exactly once.

    **Mathematics:** polynomial coefficients are estimated by least squares. R² and MAE are in-sample metrics. A chronological holdout MAE is reported separately when enough observations remain for both training and testing. Residuals equal actual minus fitted. The model's estimates at observed years are not the same thing as actual observations.

    **Function analysis:** student-entered expressions are parsed with SymPy and evaluated numerically for plotting. Derivative roots, extrema, and inflection candidates are numerical approximations. A local extremum is classified only when the first derivative changes sign; an inflection is reported only when the second derivative changes sign. Numerical scans can miss narrow features or complicated discontinuities.

    **Uncertainty:** residuals and MAE describe historical fit error. They are not validated 95% prediction intervals. No inferential interval is claimed. Polynomial extrapolation is especially risky outside the observed range and is never silently clipped.

    **Limitations:** World Bank indicators can have missing years, revisions, different coverage, and different units. A correlation or fitted curve does not establish historical causes. Custom functions are mathematical explorations, not professional engineering, medical, financial, or weather forecasts. The conceptual visual simulations do not provide clinical or safety guidance.
    """)

st.caption("For classroom use. Check the source metadata and units before interpreting any model.")
