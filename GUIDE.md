# Ghid de utilizare — Traffic Light Detection & Control

## Cuprins
1. [Cerințe și instalare](#1-cerinte-si-instalare)
2. [Structura proiectului](#2-structura-proiectului)
3. [Configurare date de test](#3-configurare-date-de-test)
4. [Rulare](#4-rulare)
5. [Interpretarea ferestrei vizuale](#5-interpretarea-ferestrei-vizuale)
6. [Parametri configurabili](#6-parametri-configurabili)
7. [Probleme frecvente și soluții](#7-probleme-frecvente-si-solutii)

---

## 1. Cerinte si instalare

### Dependențe Python

```bash
pip install ultralytics opencv-python filterpy numpy
```

### Model YOLO

Fișierul `yolov8n.pt` trebuie să existe în directorul rădăcină al proiectului.  
Dacă lipsește, rulează o dată `main.py` — Ultralytics îl descarcă automat.

> **Notă:** `yolov8n.pt` este modelul generic COCO. Funcționează rezonabil pe
> imagini rutiere reale (KITTI), dar pentru rezultate mai bune pe un simulator
> specific (CARLA, SUMO etc.) antrenează sau fine-tune-ează un model dedicat.

---

## 2. Structura proiectului

```
traffic-light-detection-and-control/
├── main.py                  # Punct de intrare — bucla principală
├── yolov8n.pt               # Ponderi YOLO (descărcate automat)
├── GUIDE.md                 # Acest ghid
└── src/
    ├── detection.py         # Detector YOLO + clasificator culoare HSV + fallback
    ├── preprocessing.py     # Blur gaussian și conversie spațiu culoare
    ├── tracking.py          # Kalman Filter pentru stabilizarea bbox-ului
    └── control.py           # Logica de decizie vehicul (GO / CAUTION / STOP)
```

---

## 3. Configurare date de test

### Opțiunea A — Dataset KITTI (implicit)

Descarcă setul de date KITTI Raw Data de la [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/) și dezarhivează-l astfel încât structura să fie:

```
data/
└── 2011_09_26/
    └── 2011_09_26_drive_0027_sync/
        └── image_02/
            └── data/
                ├── 0000000000.png
                ├── 0000000001.png
                └── ...
```

Scriptul `raw_data_downloader.sh` poate face asta automat:
```bash
bash raw_data_downloader.sh
```

### Opțiunea B — Propriile imagini sau video

Modifică în `main.py` variabila `image_folder`:

```python
image_folder = "calea/ta/catre/imagini"   # folder cu fișiere .png
```

Sau convertește un video în cadre PNG:
```bash
ffmpeg -i video.mp4 -q:v 1 frames/%05d.png
```

---

## 4. Rulare

```bash
python main.py
```

Fereastra `Traffic Light Detection - Video Simulation` se deschide și procesează imaginile una câte una.

**Comenzi în timp real:**
- `Q` — închide fereastra și oprește procesarea

---

## 5. Interpretarea ferestrei vizuale

| Element | Ce înseamnă |
|---|---|
| **Dreptunghi verde** | Bounding box al unui semafor detectat |
| **Etichetă** `Red_Light (0.87)` | Culoarea clasificată + scorul de confidence |
| **`ACTION: GO`** (verde, sus-stânga) | Vehiculul accelerează normal |
| **`ACTION: CAUTION`** (galben) | Semafor roșu/galben detectat la distanță, decelerare graduală |
| **`ACTION: STOP`** (roșu) | Oprire completă (frână aplicată) |
| **`STOP`** mare în centru | Oprire de urgență — vehiculul s-a oprit, pauză 60 secunde |
| **Telemetrie** (linia a doua) | `Throttle` / `Brake` / `Handbrake` / `Distance to Light` în metri |

### Exemplu output consolă

```
VCU DEBUG: state=Red_Light, area=312.5, size_factor=0.021, distance=18.43, confirmed=True
VCU DEBUG: state=Green_Light, area=890.0, size_factor=0.059, distance=5.21, confirmed=True
STOP
```

---

## 6. Parametri configurabili

Toți parametrii sunt în `main.py` și pot fi ajustați fără a modifica logica internă.

### Prag de confidence YOLO

```python
detections = detector.detect(blurred_frame, confidence_threshold=0.35)
```
- **0.35** (implicit) — echilibru între sensibilitate și fals-pozitive pe date KITTI
- Mărește spre `0.50` dacă vezi prea multe detecții greșite
- Coboară spre `0.20` dacă semafoarele mici/îndepărtate nu sunt detectate

### Distanța de oprire

```python
stop_distance = vcu.max_stop_distance  # implicit 5.0 metri
```
Valoarea estimată din dimensiunea bounding box-ului. Ajustează `vcu.max_stop_distance` (metri) pentru a controla cât de aproape trebuie să fie semaforul înainte de STOP complet.

### Distanța de referință (corespondență pixeli → metri)

```python
vcu.area_ref = 15000.0      # aria la care size_factor = 1.0
vcu.stop_area_pixels = 8000.0  # aria la care se declanșează zona de oprire
```
Valorile implicite sunt calibrate pentru imaginile KITTI. Pe alt dataset sau rezoluție, ajustează proporțional.

### Durata pauzei la STOP

```python
pause_duration = 60.0  # secunde
```
Cât timp rămâne afișat ecranul de STOP după oprire.

### Smoothing throttle/brake

```python
smoothing_alpha = 0.20
```
- Aproape de `0` → răspuns foarte lent (vehicul mou)
- Aproape de `1` → răspuns instant (fără smoothing)

---

## 7. Probleme frecvente si solutii

### Sistemul nu detectează niciun semafor

1. Verifică că `image_folder` din `main.py` pointează corect
2. Coboară `confidence_threshold` la `0.25`
3. Verifică că există `yolov8n.pt` în directorul rădăcină

### Prea multe detecții false (faruri, indicatoare)

1. Mărește `confidence_threshold` la `0.45` sau `0.50`
2. Dacă vine din fallback-ul HSV: ridică pragul de saturație în `_color_fallback` de la `150` la `170`

### Vehiculul se oprește la semafor verde

- A fost fix-uit: bug-ul `area > 600` a fost eliminat. Dacă se mai întâmplă, verifică în consolă linia `VCU DEBUG` — dacă `state=Green_Light` dar `ACTION=STOP`, înseamnă că `distance_to_light` e sub `max_stop_distance`. Mărește `vcu.max_stop_distance` la `3.0` sau coboară `vcu.area_ref`.

### Detecțiile "sar" sau sunt instabile

Trackerul Kalman stabilizează poziția, dar dacă vehiculul virează brusc sau semaforul iese din cadru, starea se resetează la predicție. Normal pentru un singur tracker fără re-identificare. Soluție pe termen lung: tracker multi-obiect (SORT/ByteTrack).

### `No traffic light detections in frame`

Normal pentru cadre în care nu există niciun semafor vizibil. Vehiculul continuă cu `ACTION: GO`.
