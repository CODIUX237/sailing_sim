---

# ⛵ Sailing Simulator - America's Cup RL

Questo progetto è un simulatore di regata tattica basato su **Reinforcement Learning**. L'obiettivo è addestrare un agente (la barca) a navigare tra le boe nel minor tempo possibile, gestendo la fisica del vento e il decollo sui **foil**.

## 🛠️ Requisiti di Sistema

Prima di iniziare, assicurati di avere installato:

* **Python 3.8+** (Consigliato 3.10 o 3.11 per massima compatibilità con le librerie RL).
* **FFMPEG**: Necessario per la generazione dei video `.mp4`.

## 📦 Installazione delle Dipendenze

Apri il terminale nella cartella del progetto ed esegui i seguenti comandi per installare tutti i componenti necessari:

```bash
# 1. Core Reinforcement Learning & Ambiente
pip install gymnasium stable-baselines3 shimmy

# 2. Matematica e Grafica
pip install numpy matplotlib

# 3. Generazione Video e Codec (Fondamentale!)
pip install imageio imageio[ffmpeg]

```

## 📂 Struttura del Progetto

* `main.py`: Il punto di ingresso. Gestisce l'addestramento (`train`), il test (`test`) e la registrazione (`video`).
* `environment.py`: Contiene la logica dell'ambiente Gymnasium (`SailingEnv`).
* `physics.py`: Gestisce i calcoli vettoriali del vento e la dinamica della barca.
* `render.py`: Motore grafico basato su Matplotlib per disegnare i frame della regata.
* `models/`: Cartella dove vengono salvati i modelli addestrati (file `.zip`).

## 🚀 Come Eseguire il Codice

Puoi cambiare la modalità di esecuzione modificando la variabile `MODE` all'interno del file `main.py`:

1. **Training**: Per iniziare ad addestrare l'IA da zero.
* Imposta `MODE = "train"`
* Esegui: `python main.py`


2. **Video**: Per caricare un modello esistente e generare un video `.mp4` della regata.
* Imposta `MODE = "video"`
* Esegui: `python main.py`
* Il file verrà salvato come `sailing_sim.mp4`.



## ⚠️ Risoluzione Problemi Comuni

* **Errore Video (Backend)**: Se ricevi un errore "Could not find a backend", assicurati di aver lanciato `pip install imageio[ffmpeg]`.
* **Errore Reshape (2560000)**: Se il codice crasha durante il rendering, assicurati che in `render.py` la conversione dell'immagine gestisca i 4 canali (RGBA) prima di convertirli in RGB.
* **Warning Gymnasium**: I warning relativi alla precisione dei `Box` (float64 to float32) sono normali e non bloccano l'esecuzione.

---