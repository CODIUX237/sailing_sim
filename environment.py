import gymnasium as gym
from gymnasium import spaces
import numpy as np
from render import render_frame

# Importiamo i moduli che abbiamo creato prima
from physics import WindField, SailingBoat

class AmericaCupEnv(gym.Env):
    """
    Ambiente Custom Gymnasium per la Coppa America.
    Gestisce le regole, i gate, il timer di pre-partenza e il sistema di reward.
    """
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Configurazioni di base
        self.field_size = 500.0
        self.target_radius = 20.0
        self.dt = 0.5  # Mezzo secondo per step simulato
        self.max_steps = 1000
        
        # --- SPAZIO DELLE AZIONI (CONTINUO) ---
        # action[0]: Timone da -1.0 (tutta sinistra) a 1.0 (tutta destra)
        # action[1]: Foil da 0.0 (giù) a 1.0 (su, se > 0.5 vuole volare)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # --- SPAZIO DELLE OSSERVAZIONI ---
        # Cosa vede l'agente? 
        # [x, y, heading, speed, foil_attivo, wind_speed, wind_dir, dist_target, angle_target, time_to_start]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Inizializziamo i componenti
        self.wind = None
        self.boat = None
        self.gates = []
        self.gate_index = 0
        self.current_target = None
        
        self.step_count = 0
        self.time_to_start = 120.0 # 2 Minuti di pre-partenza
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Resettiamo il Vento e la Barca
        self.wind = WindField(field_size=self.field_size)
        
        # Posizioniamo la barca nella zona di pre-partenza (es. in basso al centro)
        start_x = self.field_size / 2
        start_y = 50.0
        self.boat = SailingBoat(x=start_x, y=start_y, heading=np.pi/2)
        
        # 2. Resettiamo i contatori
        self.step_count = 0
        self.time_to_start = 120.0 # Reset timer pre-partenza
        
        # 3. Generiamo i Gate (Boe)
        # Per semplicità ora creiamo un percorso a bastone (su e giù)
        gate_up = np.array([self.field_size / 2, self.field_size - 50.0])
        gate_down = np.array([self.field_size / 2, 100.0])
        self.gates = [gate_up, gate_down, gate_up]
        self.gate_index = 0
        self.current_target = self.gates[self.gate_index]
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- 1. AGGIORNAMENTO FISICA E VENTO ---
        # Il vento evolve nel tempo (random walk)
        self.wind.step()
        
        # Otteniamo il vento specifico nel punto in cui si trova la barca
        local_wind_speed, local_wind_dir = self.wind.get_wind_at(self.boat.x, self.boat.y)
        
        # Muoviamo la barca
        action_turn = action[0]
        action_foil = action[1]
        self.boat.update_physics(self.dt, action_turn, action_foil, local_wind_speed, local_wind_dir)
        
        # --- 2. LOGICA PRE-PARTENZA (Regole) ---
        if self.time_to_start > 0:
            self.time_to_start -= self.dt
            # Se la barca supera la linea di partenza (y > 100) prima che il tempo scada (OCS)
            if self.boat.y > 100.0:
                reward -= 5.0 # Forte penalità continua per essere partiti in anticipo!
        
        # --- 3. BOUNDARY BOX (Confini del campo) ---
        # Penalità "soft", non terminiamo l'episodio ma togliamo punti
        if self.boat.x < 0 or self.boat.x > self.field_size or self.boat.y < 0 or self.boat.y > self.field_size:
            reward -= 10.0
            # Riportiamo la barca "rimbalzando" dolcemente dentro i confini
            self.boat.x = np.clip(self.boat.x, 0, self.field_size)
            self.boat.y = np.clip(self.boat.y, 0, self.field_size)

        # --- 4. CALCOLO REWARD (VMG) ---
        pos = np.array([self.boat.x, self.boat.y])
        dist_to_target = np.linalg.norm(self.current_target - pos)
        
        # Angolo verso il target
        angle_to_target = np.arctan2(self.current_target[1] - self.boat.y, self.current_target[0] - self.boat.x)
        
        # VMG: quanto la barca sta effettivamente andando verso il bersaglio
        heading_error = abs((angle_to_target - self.boat.heading + np.pi) % (2 * np.pi) - np.pi)
        vmg_to_target = self.boat.speed * np.cos(heading_error)
        
        # Diamo punti solo se il pre-start è finito, altrimenti premia restare in zona
        if self.time_to_start <= 0:
            reward += vmg_to_target * 0.1
        
            # --- 5. CONTROLLO PASSAGGIO GATE ---
            if dist_to_target < self.target_radius:
                reward += 100.0 # Grande premio per aver raggiunto la boa!
                self.gate_index += 1
                
                if self.gate_index >= len(self.gates):
                    terminated = True # Gara finita! Vittoria!
                    reward += 500.0
                else:
                    self.current_target = self.gates[self.gate_index]

        # Condizione di fine tempo massimo
        if self.step_count >= self.max_steps:
            truncated = True
            
        info = {
            'speed': self.boat.speed,
            'distance_to_target': dist_to_target,
            'time_to_start': max(0, self.time_to_start),
            'foil_active': self.boat.foil
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Costruisce l'array di osservazione per l'agente."""
        pos = np.array([self.boat.x, self.boat.y])
        dist_to_target = np.linalg.norm(self.current_target - pos)
        angle_to_target = np.arctan2(self.current_target[1] - self.boat.y, self.current_target[0] - self.boat.x)
        
        local_wind_speed, local_wind_dir = self.wind.get_wind_at(self.boat.x, self.boat.y)
        
        obs = np.array([
            self.boat.x / self.field_size,         # Normalizzato 0-1
            self.boat.y / self.field_size,         # Normalizzato 0-1
            self.boat.heading,                     # In radianti
            self.boat.speed / 40.0,                # Normalizzato rispetto a max speed
            1.0 if self.boat.foil else 0.0,        # Stato del foil
            local_wind_speed / 25.0,               # Vento locale
            local_wind_dir,                        # Dir vento locale
            dist_to_target / self.field_size,      # Distanza boa normalizzata
            angle_to_target,                       # Angolo boa
            self.time_to_start / 120.0             # Timer partenza normalizzato
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        # Generiamo il frame solo se viene richiesto
        return render_frame(self)