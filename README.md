# Practical 5 - Logging and Monitoring for an ML Model

## Project structure

| File | What it covers |
|---|---|
| `logger_config.py` | **Task 1**: logger that writes to console + rotating `logs/service.log` |
| `train_model.py` | Trains the model (Iris + RandomForest) and saves training means for drift |
| `app.py` | **Task 2**: Flask service; `/predict` logs request, prediction, latency, errors |
| `log_levels_demo.py` | **Task 3**: shows DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `monitoring.py` | **Task 4**: request/error/latency counters + summary. **Task 5**: drift check |
| `test_client.py` | Sends good, bad, and shifted requests so every log type appears |
| `evidently_report.py` | Optional Evidently drift report (HTML) |

## Setup in VS Code

2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `venv` one.

## Run it (in order)

```bash
python train_model.py          # 1. creates model/model.joblib
python log_levels_demo.py      # 2. Task 3 (INFO level - DEBUG hidden)
python log_levels_demo.py --debug   #    Task 3 again with DEBUG shown
python app.py                  # 3. start service (leave this terminal running)
```

Open a **second terminal** (click `+` in the terminal panel), activate the venv again, then:

```bash
python test_client.py          # 4. sends traffic: predictions, errors, drift
```

Then open `logs/service.log` in VS Code to see all entries.

To see the DEBUG request lines from the service, start it with:
- Windows PowerShell: `$env:LOG_LEVEL="DEBUG"; python app.py`
- macOS/Linux: `LOG_LEVEL=DEBUG python app.py`



