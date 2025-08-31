# 🚦 Traffic Tracker

A computer vision project to detect and track moving vehicles (cars, buses, bikes, etc.) in traffic videos.

---

## 🔧 Setup & Run

### Windows 🪟

```bash
cd Traffic-Tracker
cd ..
python -m venv venv
venv\Scripts\activate.bat
cd Traffic-Tracker
python -m pip install -r requirements.txt
python main.py
```

---

### Mac/Linux 🍎🐧

```bash
cd Traffic-Tracker
cd ..
python3 -m venv venv
source venv/bin/activate
cd Traffic-Tracker
pip3 install -r requirements.txt
python3 main.py
```

---

## 📌 Notes

* Requires **Python 3.8+**
* To deactivate the virtual environment:

  ```bash
  deactivate
  ```
* If OpenCV fails to install, you may need system dependencies:

  * **Ubuntu/Debian**: `sudo apt-get install python3-opencv`
  * **Mac (Homebrew)**: `brew install opencv`

---

## 🎥 Usage

1. Place your traffic video inside the project folder.
2. Update the `video_path` variable in `main.py` with your video filename.
3. Run the script to start detecting and tracking vehicles.

---

## ✅ Example Output

* Bounding boxes drawn around moving vehicles.
* Vehicle IDs tracked across frames.
* Console logs for detection and tracking progress.
