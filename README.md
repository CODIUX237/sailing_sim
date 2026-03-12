```markdown
# ⛵ America's Cup 2D - Multi-Agent RL Simulator

Un simulatore 2D avanzato di regata velica basato sulle regole della Coppa America. Il progetto utilizza **Multi-Agent Reinforcement Learning (MARL)** per addestrare due barche (agenti) a competere in un Match Race utilizzando algoritmi allo stato dell'arte (PPO via Stable-Baselines3) e la libreria PettingZoo.

## ✨ Features Principali

* **Fisica del Foiling:** Le barche possono alzare i foil per aumentare drasticamente la velocità, ma rischiano lo stallo se la velocità scende sotto la soglia critica.
* **Vento Stocastico (Random Walk):** Il campo di regata è diviso in una griglia vettoriale. Il vento cambia intensità e direzione nel tempo e nello spazio (raffiche).
* **Wind Shadow (Copertura del vento):** Modello tattico integrato. Se una barca si posiziona sopravento all'avversario, crea un "cono d'ombra" che riduce il vento (e la velocità) della barca inseguitrice.
* **Regole di Precedenza (Right of Way):** Arbitro automatico che calcola le Mure a Dritta/Sinistra (Starboard/Port) e Sottovento/Sopravento, assegnando penalità (Reward Shaping) in caso di infrazione.
* **Self-Play Training:** L'agente si allena combattendo contro le versioni precedenti di se stesso, imparando autonomamente tattiche di Match Race sempre più complesse.

## 📂 Struttura del Progetto

Il codice è modulare e diviso per responsabilità:

* `physics.py`: Motore fisico. Gestisce il drag, l'accelerazione, il foiling, la griglia del vento e il calcolo del cono d'ombra (Wind Shadow).
* `rules.py`: Motore logico ("Arbitro"). Calcola le mure e le precedenze tramite geometria vettoriale.
* `environment.py`: L'ambiente Gymnasium/PettingZoo (`AmericasCupMultiEnv`). Gestisce i reward, le penalità, i boundary del campo e i gate.
* `render.py`: Motore grafico basato su Matplotlib. Genera i frame per la visualizzazione.
* `main.py`: Entry point. Gestisce il training PPO con il Self-Play Wrapper e la generazione dei video.

## 🚀 Installazione

Assicurati di avere Python 3.9+ installato. Crea un ambiente virtuale e installa le dipendenze:

```bash
pip install gymnasium pettingzoo stable-baselines3 numpy matplotlib imageio imageio[ffmpeg]

```

## 🎮 Utilizzo

Tutta l'esecuzione è gestita tramite il file `main.py`. Apri il file e modifica la variabile `MODE` in fondo allo script:

### 1. Addestramento (Training)

Imposta `MODE = "train"` ed esegui lo script:

```bash
python main.py

```

Il modello inizierà ad addestrarsi. Al primo avvio, `boat_0` giocherà contro un avversario casuale. Ai riavvii successivi, caricherà l'ultimo modello salvato in `models/ppo_match_race.zip` e farà **Self-Play**, alzando il livello della sfida.

### 2. Generazione Video (Test)

Imposta `MODE = "video"` ed esegui lo script:

```bash
python main.py

```

Le due reti neurali si sfideranno in regata. Al termine, troverai il file `match_race.mp4` nella root del progetto.

---

## 🛠️ Troubleshooting: Il Problema del Reshape (Matplotlib)

A seconda del tuo sistema operativo o della risoluzione del monitor (DPI scaling), potresti incontrare un errore durante la generazione del video (quando `render.py` cerca di fare il reshape dell'immagine):
`ValueError: cannot reshape array of size X into shape (Y, Z, 3)`.

**La Soluzione:**
Apri il file `render.py`, vai in fondo alla funzione `render_frame`, elimina le ultime tre righe del canvas e sostituiscile con questo blocco di codice robusto:

```python
    # Converti il canvas in RGB array (Versione anti-crash per DPI)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Usiamo buffer_rgba invece di tostring_rgb (che è deprecato)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # Reshape dinamico basato sulle dimensioni REALI del buffer calcolato dal sistema
    image = buf.reshape((h, w, 4))
    # Rimuoviamo il canale Alpha (trasparenza) per tenere solo R, G, B
    image = image[:, :, :3]
    
    return image

```

Questo garantisce che le dimensioni calcolate per il video `.mp4` corrispondano sempre perfettamente alla griglia dei pixel generata da Matplotlib, indipendentemente dal tuo OS.

## 🤝 Contribuire

Ogni Pull Request è benvenuta. Se vuoi migliorare le funzioni polari in `physics.py` o aggiungere il rendering del timer di pre-partenza, sentiti libero di aprire una issue!