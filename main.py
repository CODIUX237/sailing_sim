import os
from stable_baselines3 import PPO
from environment import AmericaCupEnv
import imageio

# Creiamo le cartelle per salvare i modelli e i log se non esistono
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def train_agent():
    """Allena un nuovo agente da zero."""
    print("Inizializzazione dell'ambiente...")
    env = AmericaCupEnv()
    
    # Inizializziamo il modello PPO
    # MlpPolicy: Rete neurale standard (Multilayer Perceptron)
    # verbose=1: Stampa i progressi a schermo
    # tensorboard_log: Salva i dati per visualizzare i grafici di apprendimento
    print("Creazione del modello PPO...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOGS_DIR,
        learning_rate=0.0003,  # Standard per PPO
        batch_size=64
    )
    
    # Avviamo l'addestramento (iniziamo con 100.000 step, poi potrai alzarlo a 1-2 milioni)
    TIMESTEPS = 100000
    print(f"Inizio addestramento per {TIMESTEPS} step...")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # Salviamo il modello
    model_path = os.path.join(MODELS_DIR, "ppo_sailing_v1")
    model.save(model_path)
    print(f"Modello salvato in: {model_path}.zip")
    
    env.close()

def test_agent(model_name="ppo_sailing_v1"):
    """Carica un agente addestrato e lo fa navigare nell'ambiente."""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Errore: Il modello {model_path}.zip non esiste. Avvia prima l'addestramento!")
        return

    print(f"Caricamento del modello {model_name}...")
    model = PPO.load(model_path)
    env = AmericaCupEnv()
    
    # Testiamo il modello per 3 episodi
    episodes = 3
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        score = 0
        step_count = 0
        
        while not (done or truncated):
            # L'agente decide l'azione in base all'osservazione
            action, _states = model.predict(obs, deterministic=True)
            
            # L'ambiente esegue l'azione
            obs, reward, done, truncated, info = env.step(action)
            score += reward
            step_count += 1
            
        print(f"Episodio {ep + 1} completato in {step_count} step. Punteggio totale: {score:.2f}")
        
    env.close()

def generate_video(model_name="ppo_sailing_v1", video_name = "sailing_sim.mp4"):
    """Carica il modello e registra un video della gara. """
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path + ".zip"):
        print("Allena prima il modello!")
        return

    print("Caricamento modello per generazione video...")
    model = PPO.load(model_path)
    env = AmericaCupEnv()

    obs, info = env.reset()
    frames = []

    print("Registrazione in corso... (potrebbe volerci qualche minuto)")
    done = False
    truncated = False

    #Catturiamo il primo frame
    frames.append(env.render())

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Aggiungiamo il frame corrente alla lista
        frames.append(env.render())

        # Stampiamo un puntino per far vedere che sta lavorando
        if env.step_count % 50 == 0:
            print(".", end="", flush=True)

    print(f"\nSalvataggio del video come {video_name}...")
    # Salviamo la lista di frame come video MP4 a 30 FPS
    imageio.mimsave(video_name, frames, fps=30)
    print("Video salvato con successo!")

    env.close()

if __name__ == "__main__":
    # --- MENU DI SCELTA ---
    # Cambia la variabile qui sotto in "test" per testare il modello dopo averlo allenato
    MODE = "video"  
    
    if MODE == "train":
        train_agent()
    elif MODE == "test":
        test_agent()
    elif MODE == "video":
       generate_video()