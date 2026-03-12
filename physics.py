import numpy as np
import math

class WindField:
    """
    Gestisce un campo di vento 2D con random walk spaziale e temporale.
    Invece di avere un vento globale, divide il campo in una griglia.
    """
    def __init__(self, field_size=500, grid_resolution=50, base_speed=15.0, base_dir=np.pi/2):
        self.field_size = field_size
        self.grid_res = grid_resolution
        self.grid_size = int(field_size / grid_resolution) + 1
        
        # Inizializziamo la griglia del vento (velocità e direzione per ogni cella)
        self.wind_speed_grid = np.full((self.grid_size, self.grid_size), base_speed)
        self.wind_dir_grid = np.full((self.grid_size, self.grid_size), base_dir)
        
    def step(self):
        """Evolve il vento nel tempo aggiungendo rumore stocastico (raffiche)."""
        speed_noise = np.random.normal(0, 0.2, (self.grid_size, self.grid_size))
        dir_noise = np.random.normal(0, 0.05, (self.grid_size, self.grid_size))
        
        self.wind_speed_grid = np.clip(self.wind_speed_grid + speed_noise, 8.0, 25.0)
        
        # Normalizziamo l'angolo tra -pi e pi
        self.wind_dir_grid = (self.wind_dir_grid + dir_noise + np.pi) % (2 * np.pi) - np.pi

    def get_wind_at(self, x, y):
        """Restituisce il vento locale interpolato per una coordinata x, y."""
        # Troviamo gli indici della griglia più vicini
        grid_x = int(np.clip(x / self.grid_res, 0, self.grid_size - 1))
        grid_y = int(np.clip(y / self.grid_res, 0, self.grid_size - 1))
        
        return self.wind_speed_grid[grid_x, grid_y], self.wind_dir_grid[grid_x, grid_y]


class SailingBoat:
    """
    Gestisce la fisica, la posizione e l'attrito (drag) della barca.
    """
    def __init__(self, boat_id,  x, y, heading, max_speed=40.0):
        self.id = boat_id #per distinguere le barche
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.max_speed = max_speed
        self.foil = False
        
    def get_polar_speed(self, apparent_wind_angle, wind_speed):
        """
        Funzione polare semplificata (sostituiscila poi con la tua utils.get_polar_speed reale).
        Calcola la velocità target ideale in base all'angolo rispetto al vento.
        """
        # La barca va più veloce al traverso/lasco (circa 100-120 gradi dal vento)
        efficiency = np.sin(abs(apparent_wind_angle) / 2.0) 
        target = wind_speed * efficiency * 1.5
        return np.clip(target, 0, self.max_speed)

    def update_physics(self, dt, action_turn, action_foil, local_wind_speed, local_wind_dir):
        """
        Aggiorna la fisica in base alle azioni (continue) e al vento locale.
        - action_turn: float tra -1.0 (sx) e 1.0 (dx)
        - action_foil: boolean o float > 0.5 per indicare volontà di volo
        """
        # 1. GESTIONE TIMONE
        # Massima virata consentita: 10 gradi al secondo (moltiplicato per dt)
        max_turn_rate = np.radians(10) * dt
        turn_angle = action_turn * max_turn_rate
        self.heading = (self.heading + turn_angle + np.pi) % (2 * np.pi) - np.pi
        
        # 2. GESTIONE FOIL MANUALE CON STALLO
        wants_to_foil = action_foil > 0.5
        if wants_to_foil:
            if self.speed >= 10.0:  # Lift threshold
                self.foil = True
            elif self.speed < 8.0:  # Stall threshold
                self.foil = False
        else:
            self.foil = False
            
        # 3. CALCOLO VENTO APPARENTE
        apparent_wind_angle = (local_wind_dir - self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # 4. CALCOLO VELOCITÀ E INERZIA
        target_speed = self.get_polar_speed(apparent_wind_angle, local_wind_speed)
        
        # Moltiplicatori del Foil
        if self.foil:
            target_speed *= 1.5  # In volo andiamo più veloci
            turn_penalty = 0.8   # Virare in volo rallenta molto
        else:
            target_speed *= 0.6  # In acqua siamo lenti
            turn_penalty = 0.5   # Virare in acqua frena meno
            
        # Applichiamo la frenata causata dalla virata
        drag_penalty = turn_penalty * abs(action_turn)
        self.speed *= (1.0 - drag_penalty * dt)
        
        # Inerzia per raggiungere la target_speed
        inertia = 0.1
        self.speed = self.speed + inertia * (target_speed - self.speed)
        
        # 5. AGGIORNAMENTO POSIZIONE X, Y
        displacement = self.speed * dt
        self.x += displacement * np.cos(self.heading)
        self.y += displacement * np.sin(self.heading)

def calculate_wind_shadow(boat1, boat2, wind_speed_1, wind_speed_2, wind_dir):
    """
     Calcola l'interferenza del vento tra due barche.
    Restituisce le nuove velocità del vento tenendo conto del cono d'ombra.
    """
    dx = boat2.x - boat1.x
    dy = boat2.y - boat1.y
    dist = np.hypot(dx, dy)
        
    # Se sono a più di 100 metri, nessuna interferenza
    if dist > 100.0:
        return wind_speed_1, wind_speed_2
            
    angle_1_to_2 = np.arctan2(dy, dx)
    angle_2_to_1 = np.arctan2(-dy, -dx)
        
    # Calcoliamo verso dove SOFFIA il vento (wind_dir è da dove arriva)
    wind_down_angle = (wind_dir + np.pi) % (2 * np.pi) - np.pi
        
    # Ampiezza del cono d'ombra (es. 20 gradi a destra e sinistra)
    cone_angle = np.radians(20) 
        
    # Calcoliamo se una barca è allineata con il vento rispetto all'altra
    diff_angle_1 = abs((angle_1_to_2 - wind_down_angle + np.pi) % (2 * np.pi) - np.pi)
    diff_angle_2 = abs((angle_2_to_1 - wind_down_angle + np.pi) % (2 * np.pi) - np.pi)
        
    # Più le barche sono vicine, più i "rifiuti" sono forti (max 40% di perdita)
    shadow_factor = max(0.0, 1.0 - (dist / 100.0)) * 0.4 
        
    if diff_angle_1 < cone_angle:
        # Barca 1 è sopravento: copre la Barca 2
        wind_speed_2 *= (1.0 - shadow_factor)
    elif diff_angle_2 < cone_angle:
        # Barca 2 è sopravento: copre la Barca 1
        wind_speed_1 *= (1.0 - shadow_factor)
            
    return wind_speed_1, wind_speed_2