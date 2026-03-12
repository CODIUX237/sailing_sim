import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def render_frame(env):
    """
    Disegna l'intero campo di regata, le DUE barche, il vento e i gate.
    Restituisce l'immagine come array RGB (pronta per il video).
    """
    # Chiude figure precedenti per non saturare la RAM
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.field_size)
    ax.set_ylim(0, env.field_size)
    ax.set_facecolor('#1e3d59') # Colore dell'acqua (blu scuro)
    
    # 1. Disegna i Gate (Boe)
    for i, gate in enumerate(env.gates):
        # Controlliamo se questa boa è il target attuale di ALMENO una barca
        is_target = any(env.boat_gate_indices.get(agent, -1) == i for agent in env.agents)
        color = 'red' if is_target else 'orange'
        
        circle = plt.Circle(gate, env.target_radius, color=color, alpha=0.5)
        ax.add_patch(circle)
        ax.text(gate[0]+25, gate[1], f"Gate {i+1}", color='white')

    # 2. Disegna il Vento Globale (Freccia)
    wind_speed, wind_dir = env.wind.get_wind_at(env.field_size/2, env.field_size/2)
    ax.arrow(env.field_size - 100, env.field_size - 100, 
             40 * np.cos(wind_dir), 40 * np.sin(wind_dir),
             head_width=15, head_length=20, fc='lightblue', ec='lightblue')
    ax.text(env.field_size - 140, env.field_size - 130, f"{wind_speed:.1f} kts", color='lightblue')

    # 3. Disegna le Barche (Multi-Agent)
    # Impostiamo i colori base (in acqua) e i colori di volo (foiling)
    colors = {"boat_0": "cyan", "boat_1": "magenta"}
    foil_colors = {"boat_0": "white", "boat_1": "pink"}
    
    speed_texts = []
    
    for agent_id, boat in env.boats.items():
        boat_x, boat_y = boat.x, boat.y
        
        # Cambia colore se la barca è sui foil
        boat_color = foil_colors[agent_id] if boat.foil else colors[agent_id]
        
        # Creiamo un triangolo per indicare la direzione (heading)
        boat_length = 20
        p1 = [boat_x + boat_length * np.cos(boat.heading), boat_y + boat_length * np.sin(boat.heading)]
        p2 = [boat_x + (boat_length/2) * np.cos(boat.heading + 2.5), boat_y + (boat_length/2) * np.sin(boat.heading + 2.5)]
        p3 = [boat_x + (boat_length/2) * np.cos(boat.heading - 2.5), boat_y + (boat_length/2) * np.sin(boat.heading - 2.5)]
        
        boat_triangle = Polygon([p1, p2, p3], closed=True, color=boat_color)
        ax.add_patch(boat_triangle)
        
        # Etichetta del nome vicino alla barca (es. "boat_0")
        ax.text(boat_x + 15, boat_y - 15, agent_id, color=boat_color, fontsize=9, fontweight='bold')
        
        # Salviamo la velocità per il titolo
        speed_texts.append(f"{agent_id}: {boat.speed:.1f} kts")

    # 4. Titolo Dinamico
    title_string = f"Step: {env.step_count} | " + " | ".join(speed_texts)
    ax.set_title(title_string, color='white')

    # Rimuovi assi per un look più pulito
    ax.set_xticks([])
    ax.set_yticks([])

    # Converti il canvas in RGB array
    fig.canvas.draw() #disegna il grafico

    #usiamo rgba_buffer() che è il metodo più stabile e moderno
    #questo restituisce un array di dimensione 800*800*4 = 2.560.000
    rgba_buffer = fig.canvas.buffer_rgba()

    #Convertiamo in array numpy
    image = np.frombuffer(rgba_buffer, dtype=np.uint8)

    #reshape a 4 canali (RGBA)
    # get_width_height()[::-1] restituisce (800, 800)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # convertiamo da RGBA a RGB (Elimina il canale Alpha che causa il valueError)
    image = image[:, :, :3]
    
    return image