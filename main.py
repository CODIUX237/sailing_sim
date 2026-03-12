import os
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Importiamo i nostri moduli
from environment import AmericasCupMultiEnv
from render import render_frame

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class SelfPlayWrapper(gym.Env):
    """
    Questo Wrapper trasforma l'ambiente Multi-Agente in un ambiente Singolo per Stable-Baselines3.
    L'agente in addestramento controlla 'boat_0'.
    'boat_1' (l'avversario) è controllato dal modello salvato in precedenza (Self-Play).
    """
    def __init__(self, env, opponent_model_path=None):
        self.env = env
        # Spazi di azione e osservazione per la singola barca
        self.observation_space = env.observation_space("boat_0")
        self.action_space = env.action_space("boat_0")
        
        # Carichiamo il "fantasma" dell'avversario
        self.opponent_model = None
        if opponent_model_path and os.path.exists(opponent_model_path + ".zip"):
            print("Avversario caricato: L'agente combatterà contro una versione precedente di se stesso!")
            self.opponent_model = PPO.load(opponent_model_path)
        else:
            print("Nessun modello avversario trovato. L'avversario farà mosse casuali.")

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.current_obs = obs_dict
        return obs_dict["boat_0"], info_dict.get("boat_0", {})

    def step(self, action):
        # 1. Decidiamo l'azione dell'avversario (boat_1)
        if self.opponent_model and "boat_1" in self.current_obs:
            action_b1, _ = self.opponent_model.predict(self.current_obs["boat_1"], deterministic=True)
        else:
            action_b1 = self.action_space.sample() # Azione casuale se non c'è modello

        # 2. Assembliamo il dizionario delle azioni
        actions = {"boat_0": action}
        if "boat_1" in self.env.agents:
            actions["boat_1"] = action_b1

        # 3. Facciamo avanzare l'ambiente Multi-Agente
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.current_obs = obs

        # 4. Estraiamo solo i risultati della barca in addestramento (boat_0)
        done = terminations.get("boat_0", True)
        truncated = truncations.get("boat_0", True)
        reward = rewards.get("boat_0", 0.0)
        
        # Se la boat_0 è arrivata al traguardo, l'ambiente potrebbe non restituire più la sua obs
        obs_b0 = obs.get("boat_0", np.zeros(self.observation_space.shape, dtype=np.float32))

        return obs_b0, reward, done, truncated, infos.get("boat_0", {})


def train_marl(model_name="ppo_match_race"):
    """Allena l'agente usando il Self-Play."""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Creiamo l'ambiente base e lo "incapsuliamo" nel Wrapper per il Self-Play
    base_env = AmericasCupMultiEnv()
    env = SelfPlayWrapper(base_env, opponent_model_path=model_path)
    
    # Se esiste già un modello, continuiamo ad addestrare quello, sennò ne creiamo uno nuovo
    if os.path.exists(model_path + ".zip"):
        print("Riprendo l'addestramento del modello esistente...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creo un nuovo modello PPO da zero...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, batch_size=128)
    
    # Addestriamo per un po' di step
    TIMESTEPS = 200000
    print(f"Inizio addestramento per {TIMESTEPS} step...")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    model.save(model_path)
    print(f"Modello salvato! Ora 'boat_1' userà questa versione al prossimo addestramento.")
    env.env.close()


def generate_marl_video(model_name="ppo_match_race", video_name="match_race.mp4"):
    """Fa gareggiare il modello contro se stesso e salva il video."""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path + ".zip"):
        print("Errore: Allena prima il modello!")
        return
        
    print("Caricamento del modello per la regata...")
    model = PPO.load(model_path)
    env = AmericasCupMultiEnv()
    
    obs_dict, _ = env.reset()
    frames = [render_frame(env)]
    
    print("Registrazione in corso... (potrebbe volerci qualche minuto)")
    
    while env.agents:
        actions = {}
        # Entrambe le barche usano lo STESSO cervello per decidere cosa fare
        for agent_id in env.agents:
            action, _ = model.predict(obs_dict[agent_id], deterministic=True)
            actions[agent_id] = action
            
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        frames.append(render_frame(env))
        
        if env.step_count % 50 == 0:
            print(".", end="", flush=True)

    print(f"\nSalvataggio del video come {video_name}...")
    imageio.mimsave(video_name, frames, fps=30)
    print("Video salvato con successo!")
    env.close()

if __name__ == "__main__":
    # --- MENU DI SCELTA ---
    # Imposta "train" per addestrare, "video" per vedere la regata
    MODE = "video"  
    
    if MODE == "train":
        train_marl()
    elif MODE == "video":
        generate_marl_video()