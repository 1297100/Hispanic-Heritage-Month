# Hispanic Heritage Historical Data Explorer 2.0

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub
1. Create a GitHub repository.
2. Add `app.py`, `requirements.txt`, and this `README.md`.
3. Commit and push the files.
4. Do not commit secrets or API keys. This app uses the public World Bank API and requires no key.

## Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud and connect GitHub.
2. Choose the repository and branch.
3. Set the main file path to `app.py`.
4. Deploy. The cloud service installs dependencies from `requirements.txt`.

## Render
Create a **Python Web Service** connected to the GitHub repository.

Build command:
```bash
pip install -r requirements.txt
```

Start command (single line):
```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist
- Try population and GDP-per-capita data for Mexico and the United States.
- Change the country, indicator, year range, and polynomial degree; verify the loaded data and equation update.
- Use a degree that exceeds the number of distinct observations; the app should stop with a clear message.
- Inspect residuals and fitted values at observed years. Confirm residual = actual − fitted.
- Compare single-year and batch estimates for the same year; they should agree because both use the same centered-time transformation.
- Enter malformed custom notation, `log(x)`, and a rational expression such as `1/(x-2000)`; confirm parse errors or graph gaps are handled.
- Test invalid custom batch entries and range steps 1–10.
- Change the function, context, units, course mode, analysis interval, point, and secant interval.
- Test Play, Pause, Reset, and speed controls.
- Download and open CSV, `.xlsx`, and PDF. Check the workbook has Summary, Working data, and Calculated years sheets.
- Test a country comparison and inspect missing-year gaps.
- Deploy, open the public URL, then inspect Render / Streamlit logs if the app fails.

## Common errors
- **No data:** the selected indicator may not be reported for the country and years chosen.
- **API / timeout error:** check the World Bank API, network access, and deployment logs; retry later.
- **Dependency error:** confirm the build log installed every package in `requirements.txt`.
- **Port binding error on Render:** use the exact `$PORT` start command above.
- **Blank graph / undefined values:** inspect the function domain. Logarithms need positive arguments; denominators cannot be zero.
- **Unusual polynomial predictions:** high-degree fits can oscillate and extrapolate badly. Do not treat extrapolated values as facts.

## Important limitations
The fit is descriptive, not causal. R² and MAE are in-sample; holdout MAE is a separate chronological test. Neither is a prediction interval. Derivative analysis is numerical for root finding and may miss narrow behavior. The app does not provide validated medical, engineering, or financial advice.
