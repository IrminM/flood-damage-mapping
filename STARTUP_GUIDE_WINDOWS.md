# FloodIntel — Windows Quickstart & Troubleshooting Guide

This guide ensures any teammate can extract, configure, and run the FloodIntel platform on a fresh Windows computer without experiencing connection or dependency errors.

---

## 1. What Does `http://127.0.0.1:8000` Mean?

* `127.0.0.1` is the standard loopback IP address representing **localhost** (the local computer you are currently using).
* Port `8000` is where the **FloodIntel FastAPI backend** listens for requests from the browser dashboard.
* If you see the message:  
  `Assessment Failed: Unable to connect to backend service at http://127.0.0.1:8000/api/assess`  
  **it means your browser cannot find the backend server because it has not been started yet or its terminal was closed.** The dashboard in your browser needs the local backend service running in the background to fetch flood layers and compute metrics.

---

## 2. Quickstart Startup (Two Easy Options)

### Option A: 1-Click Launcher (Recommended for Windows)

1. Extract the downloaded project ZIP to a local folder (e.g., `C:\FloodIntel\floodintel-track-b`).
2. Double-click **`run_dashboard.bat`** in the extracted folder.
   - It will automatically create `.venv`, install dependencies from `requirements.txt`, launch the FastAPI server, and open your web browser to [http://127.0.0.1:8000/floodintel/](http://127.0.0.1:8000/floodintel/).
3. **Keep the Command Prompt window open** while using the dashboard.

---

### Option B: Manual Command-Line Startup (Step-by-Step)

#### Step 1: Open Terminal in the Project Directory
Open **Command Prompt (`cmd`)** or **PowerShell** and navigate to your extracted folder:
```cmd
cd C:\path\to\floodintel-track-b
```

#### Step 2: Create and Activate Virtual Environment
```cmd
:: Using Command Prompt (cmd.exe):
python -m venv .venv
.\.venv\Scripts\activate.bat
```
```powershell
# Using PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### Step 3: Install Required Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```
*(For development and testing, you can also run `pip install -r requirements-dev.txt`).*

#### Step 4: Start the FastAPI Backend Server
```cmd
python scripts/run_server.py
```
You will see:
```text
==============================================================
 Starting Flood Damage Mapper Backend (FastAPI)
 Dashboard:    http://127.0.0.1:8000/floodintel/
 API Base:     http://127.0.0.1:8000/api
 Swagger UI:   http://127.0.0.1:8000/docs
 Health Check: http://127.0.0.1:8000/api/v1/health
 Note:         127.0.0.1 refers to this local computer running the browser.
               Keep this terminal window open while using the dashboard.
==============================================================
```

#### Step 5: Open the Interactive Dashboard
Open your web browser (Edge, Chrome, Firefox) and navigate to:
**[http://127.0.0.1:8000/floodintel/](http://127.0.0.1:8000/floodintel/)**

Click **▶ Run Flood Assessment** to load the validated multi-sensor hazard corridor, infrastructure impact counts, cut-off settlement analysis, and downstream drainage path.

---

## 3. Troubleshooting Common Issues

### Issue 1: "Assessment Failed: Unable to connect to backend service"
* **Cause:** The backend server (`run_server.py`) is either not running, crashed on startup, or the terminal window running it was closed.
* **Fix:** Open a terminal in the project directory, run `python scripts/run_server.py`, and verify you see `Application startup complete`. **Keep that terminal open**.

### Issue 2: Port 8000 is already in use
* **Cause:** Another local program or a previous FloodIntel process is using port 8000.
* **Fix:** Launch the backend on port 8080 instead:
  ```cmd
  python scripts/run_server.py --port 8080
  ```
  Then open [http://127.0.0.1:8080/floodintel/](http://127.0.0.1:8080/floodintel/) in your browser.

### Issue 3: Incomplete Extraction from ZIP
* **Cause:** Running files directly inside Windows Explorer's compressed zip preview without extracting, or antivirus quarantine.
* **Fix:** Right-click the `.zip` file $\rightarrow$ **Properties** $\rightarrow$ check **Unblock** if present $\rightarrow$ click **Extract All...** to an uncompressed directory before running.

### Issue 4: Running Tests
To verify all unit tests on your machine:
```cmd
pytest
```
All unit tests should pass with 0 failures.

---

## 4. Evaluation Scope & Mandatory Attributions

* **Operational Scope:** Verified and fully prepared for the **Trishuli River Basin (August 2026)**. External case studies (Emilia-Romagna, Valencia) require local data ingestion and report `Data Not Prepared` with zero cross-corridor leakage. Arbitrary global live downlink queries require external cloud credentials and are rejected with descriptive HTTP 400 responses.
* **Mandatory Legal Attributions:**
  * Contains modified Copernicus Sentinel data 2026.
  * Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
  * Sen1Floods11 dataset (Bonafilia et al., 2020) HandLabeled baseline.
  * Copernicus Emergency Management Service Rapid Mapping (EMSR927).
