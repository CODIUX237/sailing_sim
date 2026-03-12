import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Importiamo i nostri moduli di fisica e regole
from physics import WindField, SailingBoat, calculate_wind_shadow
from rules import check_penalties

class AmericasCupMultiEnv(ParallelEnv):
    """
    Ambiente Multi-Agente per il Match Race della Coppa America.
    Supporta 2 barche che competono applicando regole di precedenza e wind shadow.
    """
    metadata = {"render_modes": ["rgb_array"], "name": "americas_cup_marl_v1"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Definiamo gli agenti
        self.possible_agents = ["boat_0", "boat_1"]
        self.agents = self.possible_agents.copy()
        
        self.field_size = 500.0
        self.target_radius = 20.0
        self.dt = 0.5
        self.max_steps = 1000
        
        # Dizionari per spazi di azione e osservazione
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for agent in self.possible_agents:
            # Action: [Timone (-1 a 1), Foil (0 a 1)]
            self.action_spaces[agent] = spaces.Box(
                low=np.array([-1.0, 0.0]), 
                high=np.array([1.0, 1.0]), 
                dtype=np.float32
            )
            
            # Obs: 13 valori (5 propri + 2 vento + 2 target + 4 nemico)
            self.observation_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )
            
        self.wind = None
        self.boats = {}
        self.gates = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.step_count = 0
        
        self.wind = WindField(field_size=self.field_size)
        
        # Posizioniamo le barche separate sulla linea di partenza
        self.boats = {
            "boat_0": SailingBoat(boat_id="boat_0", x=self.field_size/2 - 50, y=50.0, heading=np.pi/2),
            "boat_1": SailingBoat(boat_id="boat_1", x=self.field_size/2 + 50, y=50.0, heading=np.pi/2)
        }
        
        # Gate a bastone
        gate_up = np.array([self.field_size / 2, self.field_size - 50.0])
        gate_down = np.array([self.field_size / 2, 100.0])
        self.gates = [gate_up, gate_down, gate_up]
        
        # Indici dei gate per ogni barca
        self.boat_gate_indices = {"boat_0": 0, "boat_1": 0}
        
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.step_count += 1
        
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # 1. EVOLUZIONE VENTO
        self.wind.step()
        
        # Vento al centro del campo (utile per le regole generali)
        _, global_wind_dir = self.wind.get_wind_at(self.field_size/2, self.field_size/2)

        # 2. CALCOLO WIND SHADOW (Fisica condivisa)
        b0, b1 = self.boats["boat_0"], self.boats["boat_1"]
        w_speed_0, w_dir_0 = self.wind.get_wind_at(b0.x, b0.y)
        w_speed_1, w_dir_1 = self.wind.get_wind_at(b1.x, b1.y)
        
        # Applichiamo i rifiuti prima di muovere le barche
        w_speed_0, w_speed_1 = calculate_wind_shadow(b0, b1, w_speed_0, w_speed_1, global_wind_dir)

        # 3. AGGIORNAMENTO FISICA SINGOLE BARCHE
        for agent in self.agents:
            if agent in actions:
                action = actions[agent]
                boat = self.boats[agent]
                wind_spd = w_speed_0 if agent == "boat_0" else w_speed_1
                wind_dir = w_dir_0 if agent == "boat_0" else w_dir_1
                
                boat.update_physics(self.dt, action[0], action[1], wind_spd, wind_dir)
                
                # Penalità Uscita dal Campo
                if boat.x < 0 or boat.x > self.field_size or boat.y < 0 or boat.y > self.field_size:
                    rewards[agent] -= 5.0
                    boat.x = np.clip(boat.x, 0, self.field_size)
                    boat.y = np.clip(boat.y, 0, self.field_size)

        # 4. CONTROLLO REGOLE E COLLISIONI (Arbitro)
        pen_0, pen_1 = check_penalties(b0, b1, global_wind_dir)

        #applica la penalità solo se la barca è ancora nel dizionnario dei rewards
        if "boat_0" in rewards:
            rewards["boat_0"] += pen_0
        if "boat_1" in rewards:
            rewards["boat_1"] += pen_1

        # 5. PROGRESSO E VMG (Reward Shaping)
        for agent in list(rewards.keys()): #usiamo list per evitare errori di iterazioni se eliminiamo agenti
            boat = self.boats[agent]
            target = self.gates[self.boat_gate_indices[agent]]
            
            dist_to_target = np.linalg.norm(target - np.array([boat.x, boat.y]))
            angle_to_target = np.arctan2(target[1] - boat.y, target[0] - boat.x)
            
            # VMG verso la boa
            heading_error = abs((angle_to_target - boat.heading + np.pi) % (2 * np.pi) - np.pi)
            vmg = boat.speed * np.cos(heading_error)
            rewards[agent] += vmg * 0.1
            
            # Passaggio Boa
            if dist_to_target < self.target_radius:
                rewards[agent] += 50.0
                self.boat_gate_indices[agent] += 1
                
                # Se ha finito il percorso
                if self.boat_gate_indices[agent] >= len(self.gates):
                    terminations[agent] = True
                    rewards[agent] += 200.0 # Premio vittoria

        # Condizione di fine tempo (Truncation)
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True

        # Rimuoviamo gli agenti che hanno finito
        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_id):
        """Costruisce l'osservazione includendo i dati del nemico."""
        boat = self.boats[agent_id]
        enemy_id = "boat_1" if agent_id == "boat_0" else "boat_0"
        enemy = self.boats[enemy_id]
        
        target = self.gates[min(self.boat_gate_indices[agent_id], len(self.gates)-1)]
        dist_to_target = np.linalg.norm(target - np.array([boat.x, boat.y]))
        angle_to_target = np.arctan2(target[1] - boat.y, target[0] - boat.x)
        
        local_wind_spd, local_wind_dir = self.wind.get_wind_at(boat.x, boat.y)
        
        # Calcoliamo posizione relativa del nemico
        rel_enemy_x = enemy.x - boat.x
        rel_enemy_y = enemy.y - boat.y
        
        obs = np.array([
            boat.x / self.field_size,
            boat.y / self.field_size,
            boat.heading,
            boat.speed / 40.0,
            1.0 if boat.foil else 0.0,
            local_wind_spd / 25.0,
            local_wind_dir,
            dist_to_target / self.field_size,
            angle_to_target,
            # --- Dati Nemico ---
            rel_enemy_x / self.field_size,
            rel_enemy_y / self.field_size,
            enemy.heading,
            enemy.speed / 40.0
        ], dtype=np.float32)
        
        return obs
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]