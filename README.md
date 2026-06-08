# Traffic Light Detection — Computer Vision Clasic (fără ML)

Detectează semafoare **roșu** și **verde** în cadre de cameră, folosind doar OpenCV (threshold HSV, morfologie, componente conexe, filtre geometrice). Fără YOLO, fără modele antrenate, fără date de training.

Tunat pe [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset).

---

## Cuprins

1. [Structura proiectului](#structura)
2. [`main.py` — interfața CLI](#mainpy)
3. [`src/detector.py` — motorul de detecție](#detectorpy)
4. [`src/tracker.py` — stabilizare temporală](#trackerpy)
5. [`src/evaluate.py` — evaluare vs ground truth](#evaluatepy)
6. [`src/visualize.py` — desenare casete](#visualizepy)
7. [Pipeline complet (end-to-end)](#pipeline)
8. [Reglarea pragurilor](#reglare)
9. [Utilizare](#utilizare)

---

## <a name="structura"></a>1. Structura proiectului

```
main.py                CLI: image / folder / play / eval
src/
  detector.py          Detecția propriu-zisă (HSV + morfologie + filtre)
  tracker.py           Stabilizare temporală între cadre
  evaluate.py          Compară detecții cu CSV-ul LISA
  visualize.py         Desenează casete pe cadre
requirements.txt       opencv-python, numpy
data/                  Dataset-ul LISA (cadre + frameAnnotationsBOX.csv)
output/                Cadre adnotate (creat la rulare)
```

---

## <a name="mainpy"></a>2. `main.py` — interfața CLI

Punctul de intrare. Are 4 subcomenzi.

### Funcții

#### `_ensure_out_dir()`
Creează directorul `output/` dacă nu există.

#### `_list_frames(folder)`
Returnează lista sortată a fișierelor `.jpg/.jpeg/.png` dintr-un director.

#### `cmd_image(path)`
Detectează semafoare într-o singură imagine.
- Citește imaginea (`cv2.imread`)
- Apelează `detect(img)`
- Tipărește detecțiile la consolă
- Salvează imaginea adnotată în `output/<nume>_det.jpg`

#### `cmd_folder(folder, limit=None)`
Procesează toate cadrele dintr-un folder.
- Iterează prin cadre
- Detectează în fiecare
- Salvează rezultatele în `output/<nume_folder>/`
- **Parametru `limit`**: oprește după N cadre (pentru testare rapidă)

#### `_resolve_play_paths(path)`
Helper pentru `cmd_play`: acceptă fie un folder de cadre, fie un folder de clip LISA (care conține `frames/` + `frameAnnotationsBOX.csv`). În al doilea caz, încarcă și ground truth-ul pentru afișare.

#### `_hud(img, text, y=24)`
Desenează un overlay text în colțul stâng-sus al cadrului (nume frame, fps, număr detecții etc).

#### `cmd_play(path, fps, loop, show_gt, scale, smooth)`
Redă cadrele într-o fereastră OpenCV ca un video, cu detecții în timp real.
- **`fps`**: viteza-țintă de redare (cadre/secundă)
- **`loop`**: revine la primul cadru după ultimul
- **`show_gt`**: afișează casetele ground truth (cyan/galben) dacă există CSV
- **`scale`**: factor de scalare al ferestrei (0.75 = mai mică, 1.5 = mai mare)
- **`smooth`**: activează tracker-ul temporal (vezi `tracker.py`)

Taste:
- `space` — pauză
- `n`/`p` — frame următor/anterior
- `+`/`-` — accelerează/încetinește
- `g` — toggle ground truth
- `s` — salvează snapshot
- `q`/`ESC` — ieșire

#### `cmd_eval(clip_dir, iou_threshold, save_examples, limit)`
Evaluează detectorul pe un clip cu CSV de adnotări.
- **`iou_threshold`**: prag IoU pentru a considera o detecție = TP (uzual 0.2-0.5)
- **`save_examples`**: numărul de cadre adnotate de salvat pentru inspecție vizuală
- **`limit`**: oprește după N cadre
- Tipărește: TP, FP, FN, Precision, Recall, F1

#### `main(argv)`
Parsează argumente cu `argparse`. Dispatchează spre `cmd_image/folder/play/eval`.

---

## <a name="detectorpy"></a>3. `src/detector.py` — motorul de detecție

Aici se întâmplă tot ce e important. Pipeline-ul logic:

```
BGR --convert--> HSV
   --threshold--> mask_red, mask_green
   --morfologie--> măști curățate
   --componente conexe--> blob-uri candidat
   --filtre geometrice--> candidați validați
   --filtre context (dark surround, above-dark)--> validați
   --reject brake-pairs--> rămas roșu
   --expand box (bec → carcasă)--> casete finale
   --NMS--> detecții
```

### Tunabile (parametri de reglare)

#### Praguri culoare — măștile HSV

```python
RED_STRICT_LO_1 = [0,   70, 150]   # Hue 0-8, S>=70, V>=150
RED_STRICT_HI_1 = [8,  255, 255]
RED_STRICT_LO_2 = [172, 70, 150]   # Hue se rotește pe 0/180 -> 2 intervale
RED_STRICT_HI_2 = [180, 255, 255]

RED_WIDE_LO_1   = [0,   70, 150]   # Hue 0-18 (extinde spre portocaliu pentru apus)
RED_WIDE_HI_1   = [18, 255, 255]
RED_WIDE_LO_2   = [160, 70, 150]
RED_WIDE_HI_2   = [180, 255, 255]

GREEN_LOWER = [40,  60, 150]       # Hue 40-95
GREEN_UPPER = [95, 255, 255]
```

- **STRICT** = nucleul roșu pur (LED real saturează aici)
- **WIDE** = include halo-ul portocaliu de la apus
- Doar pixelii WIDE care au un pixel STRICT în vecinătate sunt păstrați (anchor-and-grow) — felinarele de stradă pur-portocalii nu trec
- `RED_ANCHOR_DILATE = 3` — vecinătatea (3×3) în care un pixel WIDE caută o ancoră STRICT
- `RED_WIDE_MAX_Y = 0.32` — masca WIDE se aplică doar în top 32% al cadrului (semafoarele atârnă sus; felinarele nu)

#### Filtre geometrice pe blob-uri

```python
MIN_AREA          = 8       # minim 8 pixeli -> filtrează speckle
MAX_AREA_FRAC     = 0.02    # max 2% din imagine -> filtrează blob-uri uriașe
MIN_ASPECT        = 0.5     # raport lățime/înălțime > 0.5
MAX_ASPECT        = 1.6     # raport < 1.6 (becul e ~circular, nu alungit orizontal)
MIN_EXTENT        = 0.5     # area / (w*h) > 0.5 -> compact, nu stringy
VERTICAL_MAX_FRAC = 0.48    # centrul becului < 48% înălțime (sub asta -> car/sol)
```

Pe 1152 ground truth-uri din LISA day, max cy/H = 0.46. Deci 0.48 e cu o marjă de siguranță.

#### Verificare „dark surround"

```python
DARK_SURROUND_VMAX = 95     # V mediu al inelului din jurul becului < 95
```

Un semafor real are carcasă neagră în jurul becului. Un far de mașină pe caroseria deschisă nu are. Scorul `ds` e între 0 (inel luminos) și 1 (inel întunecat).

#### Verificare „above is dark" (anti-brake-light)

```python
ABOVE_DARK_VMAX     = 110   # V mediu al coloanei deasupra becului < 110
ABOVE_HEIGHT_FACTOR = 1.5   # coloana e de 1.5 * h_bec înălțime
ABOVE_CHECK_MIN_Y   = 0.30  # verificarea se aplică doar dacă cy/H > 0.30
```

Un semafor are carcasă neagră deasupra (becurile galben/verde stinse). Un far de mașină de obicei nu — are cerul, mașini, sol mai luminos deasupra.

#### Respingere perechi de farun

```python
PAIR_Y_TOL           = 0.5    # |dy|/h < 0.5 = "la aceeași înălțime"
PAIR_SIZE_TOL        = 0.6    # diferența relativă de mărime < 0.6
PAIR_MIN_DIST_FACTOR = 1.5    # distanța între farun > 1.5 * h
PAIR_MAX_DIST_FACTOR = 25.0   # distanța între farun < 25 * h
```

Două blob-uri roșii cu aceeași înălțime și mărime, separate orizontal pe o distanță plauzibilă de lățime de mașină → presupus far de mașină → ambele șterse.

#### Override pentru becuri în cer (apus, fundal luminos)

```python
SKY_BULB_MAX_Y      = 0.25   # cy/H < 0.25 = bec sus în cer
SKY_BULB_MIN_MEAN_V = 215    # impune V mediu mare în interiorul becului
SKY_BULB_MIN_MEAN_S = 170    # impune S mediu mare în interiorul becului
```

În top 25% al cadrului, „dark surround" e inutil (cerul e luminos). În schimb cer becul în sine să fie strălucitor și foarte saturat — LED-urile reale satisfac asta; reflexele difuze pe frunze nu.

### Funcții

#### `_red_mask(hsv)`
Returnează masca de roșu folosind anchor-and-grow:
1. Calculează masca STRICT (H ≤ 8, S ≥ 70, V ≥ 150) globală
2. Calculează masca WIDE (H ≤ 18) globală
3. Dilată STRICT cu o vecinătate 3×3 → halo
4. WIDE ∩ halo = pixeli portocalii lângă un nucleu roșu
5. Output = STRICT global + (WIDE ∩ halo) doar în top 32%

Astfel:
- Lumini roșii pure (orice cadru) trec prin STRICT
- Lumini „roșu→portocaliu" la apus (sus în cadru) trec prin WIDE cu ancoră
- Felinare pur portocalii (chiar sus) NU trec (n-au ancoră STRICT)

#### `_green_mask(hsv)`
Threshold simplu HSV pentru verde (H 40-95, S ≥ 60, V ≥ 150).

#### `_clean(mask)`
Curățare morfologică: open (3×3 elipsă, 1 iter) → close (3×3, 2 iter). Elimină speckle și umple găuri mici în bec.

#### `_dark_surround_score(v, x, y, w, h)`
Returnează scor 0-1: cât de întunecat e inelul în jurul becului. Mostră: un dreptunghi de 3× dimensiunea becului în jurul lui, exclude becul, calculează V mediu pe inel. Scor = `1 - mean_V/DARK_SURROUND_VMAX`, clamped 0-1.

#### `_above_mean_v(v, x, y, w, h)`
V mediu pe o coloană îngustă imediat deasupra becului. Folosit pentru a verifica că deasupra e carcasă neagră (semafor) sau cer luminos (far).

#### `_filter_blobs(mask, hsv, label, frame_shape)`
Aplică pe fiecare blob (componentă conexă) toate filtrele:
1. Arie [MIN_AREA, MAX_AREA_FRAC×H×W]
2. Aspect ratio [MIN_ASPECT, MAX_ASPECT]
3. Extent ≥ MIN_EXTENT
4. cy/H ≤ VERTICAL_MAX_FRAC
5. **Dacă cy/H < SKY_BULB_MAX_Y**: V_intern ≥ 215 ȘI S_intern ≥ 170
6. **Altfel**: dark_surround ≥ 0.2
7. **Dacă roșu și cy/H > 0.30**: V deasupra ≤ 110
8. Calculează scor:
   - `extent_score = 0.5 * extent + 0.5 * dark_surround`
   - `vfac = 1.0 - 0.45 * (cy/H) / VERTICAL_MAX_FRAC` (becuri sus = boost, jos = discount)
   - `score = extent_score * vfac`

Returnează lista de `Detection`.

#### `_reject_brake_pairs(reds)`
Iterează prin perechi de blob-uri roșii. Dacă două sunt la aceeași y, aceeași mărime și separate pe distanță de mașină → ambele respinse.

#### `_expand(d, factor, W, H)`
Mărește caseta becului (4×4 px) la o casetă „carcasă de semafor" (≈ 3× lățime, 4× înălțime):
```python
dx = w * factor              # +factor*w fiecare parte
dy = h * factor * 1.5        # carcasă mai înaltă decât lată
```

#### `_iou(a, b)` și `_nms(dets, iou_threshold)`
Standard: Intersection-over-Union și Non-Maximum Suppression. NMS elimină casete duplicate (ex: pixeli STRICT + WIDE care formează două componente suprapuse pe același bec).

#### `detect(frame_bgr, expand=1.0)` — **API public**
1. BGR → HSV
2. Calculează mask_red și mask_green
3. Curățare morfologică
4. Filtrare blob-uri pentru ambele
5. Respinge perechi de farun (doar roșu)
6. Expandează casetele la dimensiunea carcasei
7. NMS

Returnează `List[Detection]`.

#### `dataclass Detection`
```python
x1, y1, x2, y2: int      # casetă bounding
label: str               # "red" sau "green"
score: float             # 0-1, încrederea
```

---

## <a name="trackerpy"></a>4. `src/tracker.py` — stabilizare temporală

Detectorul lucrează cadru-cu-cadru și produce flicker (un cadru detectează, următorul nu). Tracker-ul stabilizează.

### `class Tracker`

#### Constructor: `Tracker(min_hits=2, max_age=3, match_dist=40)`
- **`min_hits`**: o detecție trebuie văzută în atâtea cadre înainte de a fi raportată
- **`max_age`**: o detecție confirmată rămâne afișată atâtea cadre după ce dispare
- **`match_dist`**: prag de distanță în pixeli pentru a asocia o detecție cu o pistă existentă

#### `update(dets) -> List[Detection]`
Apelat la fiecare cadru cu detecțiile brute. Logica:
1. Pentru fiecare detecție, găsește cea mai apropiată pistă existentă (același label, distanță < `match_dist`)
2. Actualizează pista (poziție, `last_seen_frame`, incrementează `hits`)
3. Detecțiile neasociate creează piste noi (cu `hits=1`)
4. Pistele care n-au fost actualizate de mai mult de `max_age` cadre sunt șterse
5. Returnează doar pistele cu `hits ≥ min_hits`

#### `reset()`
Golește toate pistele. Apelat când utilizatorul sare la alt cadru cu `p`.

### `dataclass _Track` (intern)
```python
cx, cy: float            # centru curent
label: str               # "red" sau "green"
hits: int                # de câte ori a fost văzută
last_seen_frame: int     # ultimul cadru când a apărut
last_det: Detection      # ultima casetă pentru afișare
```

### De ce ajută
- **Filtrează FP-uri tranzitorii**: o reflexie de frunziș apare 1 cadru → nu confirmă (`hits=1 < min_hits`)
- **Acoperă flicker-ul**: un semafor confirmat dispare 1 cadru → încă afișat (`age < max_age`)

---

## <a name="evaluatepy"></a>5. `src/evaluate.py` — evaluare vs ground truth

Compară detecțiile cu CSV-ul LISA.

### Format CSV LISA
```
Filename;Annotation tag;Upper left X;Upper left Y;Lower right X;Lower right Y;...
```
Tag-uri:
- `stop*` → roșu
- `go*` → verde
- `warning*` → ignorat (galben nu e detectat de noi)

### Funcții

#### `_tag_to_label(tag)`
Convertește tag-ul brut LISA în „red"/„green" sau `None`.

#### `load_annotations(csv_path)`
Parsează CSV-ul. Returnează `{nume_frame: [GroundTruth, ...]}`.

#### `_iou(a, b)`
IoU pentru două tuple `(x1, y1, x2, y2)`.

#### `evaluate_frame(dets, gts, iou_threshold=0.3)`
Matching greedy per clasă: sortează detecțiile după scor (descendent), fiecare detecție găsește GT-ul cu cel mai mare IoU (același label, neasociat încă).
- IoU ≥ prag → TP
- IoU < prag → FP
- GT-uri rămase neasociate → FN

Returnează `FrameResult(tp, fp, fn)`.

#### `class Summary`
Acumulator pentru toate cadrele.
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2*P*R / (P+R)`

### `dataclass GroundTruth`
```python
x1, y1, x2, y2: int      # din CSV
label: str               # "red" sau "green"
```

### `dataclass FrameResult`
```python
tp, fp, fn: int          # contoare per cadru
```

---

## <a name="visualizepy"></a>6. `src/visualize.py` — desenare casete

Funcții de redare.

#### `draw_detections(frame_bgr, dets, thickness=2, show_score=True)`
Desenează casete colorate (roșu pentru „red", verde pentru „green") cu etichetă + scor deasupra fiecăreia.

#### `draw_ground_truth(frame_bgr, gts, thickness=1)`
Desenează casetele ground truth în cyan/galben (intentionat diferite de detecții ca să poată fi comparate vizual). Etichetă „gt:red"/„gt:green".

---

## <a name="pipeline"></a>7. Pipeline complet (end-to-end)

Un cadru ajunge în `detect()`. Pas cu pas:

1. **Convert BGR → HSV** (`cv2.cvtColor`).
2. **Masca de roșu** via anchor-and-grow:
   - Strict (H≤8): găsește nucleele LED
   - Wide (H≤18): găsește halo-ul portocaliu (doar top 32%)
   - Output = Strict + WIDE_ancorat
3. **Masca de verde** simplă: `cv2.inRange` pe intervalul verde.
4. **Curățare morfologică**: open + close → blob-uri compacte.
5. **Componente conexe** (`cv2.connectedComponentsWithStats`).
6. **Pentru fiecare blob**:
   - Filtre geometrice (arie, aspect, extent, vertical)
   - Branch sky-bulb (top 25%): cer V_intern, S_intern mari
   - Branch ground: cer dark surround
   - Verificare anti-brake-light pentru roșu (above is dark)
   - Calcul scor cu pondere verticală
7. **Reject brake pairs** pe blob-urile roșii (perechi → înlăturate).
8. **Expand box** bec → carcasă (≈3× lățime, 4× înălțime).
9. **NMS** pentru a elimina suprapuneri.
10. Returnează lista finală.

În viewer (`cmd_play`), rezultatul trece în `Tracker.update()` pentru stabilizare temporală înainte de afișare.

---

## <a name="reglare"></a>8. Reglarea pragurilor — sfaturi

Toți parametrii sunt în vârful `src/detector.py`. Reguli generale:

| Vrei | Modifică |
|---|---|
| Mai puține FP din mașini | Coboară `VERTICAL_MAX_FRAC` (ex. 0.45) |
| Mai puține FP din felinare orange | Strânge STRICT (`H≤6` în loc de `H≤8`) sau scade `RED_WIDE_MAX_Y` |
| Mai multe lumini prinse la apus | Lărgește WIDE (`H≤20`) sau crește `RED_WIDE_MAX_Y` |
| Mai puține FP în frunze (apus) | Crește `SKY_BULB_MIN_MEAN_V` și `_MIN_MEAN_S` |
| Flicker între cadre | Crește `Tracker(max_age=5)` |
| Detecții false instabile | Crește `Tracker(min_hits=3)` |

---

## <a name="utilizare"></a>9. Utilizare

```bash
pip install -r requirements.txt

# O singură imagine
python main.py image data/sample-dayClip6/sample-dayClip6/frames/dayClip6--00198.jpg

# Folder de cadre → rezultate în output/
python main.py folder data/sample-dayClip6/sample-dayClip6/frames --limit 50

# Redare ca video, cu detecții live și tracker
python main.py play data/sample-dayClip6/sample-dayClip6 --fps 20

# Evaluare vs CSV
python main.py eval data/sample-dayClip6/sample-dayClip6
```

### Metrici curente (pe primele 200 cadre)

| | Day | Night |
|---|---|---|
| Precision | 0.78 | 0.74 |
| Recall    | 0.53 | 0.80 |
| F1        | 0.63 | 0.77 |

### Limitări fundamentale (fără ML)

- **Felinare portocalii noaptea** pe carcasă neagră — au aceeași semnătură HSV+context ca un semafor roșu LED. Pot fi reduse cu tracker (filtru tranzitorii) dar nu eliminate complet.
- **Apus în contralumină extremă** — semaforul devine silueta cu inel luminos, depinde dacă păstrează măcar câțiva pixeli H ≤ 8 în miez.
- **Far singular de mașină în față** — fără pereche și pe carcasă întunecată, nu poate fi distins de un semafor roșu mic la distanță.

Pentru aceste cazuri, soluția corectă e ML supervizat (YOLO, etc.) sau temporal motion (cum se mișcă obiectul între cadre).
