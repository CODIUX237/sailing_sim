import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def render_frame(env):
    """
    Disegna l'intero campo di regata, la barca, il vento e i gate.
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
        color = 'red' if i == env.gate_index else 'orange'
        circle = plt.Circle(gate, env.target_radius, color=color, alpha=0.5)
        ax.add_patch(circle)
        ax.text(gate[0]+25, gate[1], f"Gate {i+1}", color='white')

    # 2. Disegna la Linea di Partenza (Start Line)
    ax.axhline(y=100.0, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    if env.time_to_start > 0:
        ax.text(10, 105, f"START IN: {env.time_to_start:.1f}s", color='yellow', fontsize=12, fontweight='bold')

    # 3. Disegna il Vento Locale (Freccia)
    # Prendiamo il vento al centro del campo per la visualizzazione generale
    wind_speed, wind_dir = env.wind.get_wind_at(env.field_size/2, env.field_size/2)
    ax.arrow(env.field_size - 100, env.field_size - 100, 
             40 * np.cos(wind_dir), 40 * np.sin(wind_dir),
             head_width=15, head_length=20, fc='lightblue', ec='lightblue')
    ax.text(env.field_size - 140, env.field_size - 130, f"{wind_speed:.1f} kts", color='lightblue')

    # 4. Disegna la Barca
    boat_x, boat_y = env.boat.x, env.boat.y
    boat_color = 'cyan' if env.boat.foil else 'white' # Cambia colore se vola!
    
    # Creiamo un semplice triangolo per indicare la direzione (heading)
    boat_length = 20
    p1 = [boat_x + boat_length * np.cos(env.boat.heading), boat_y + boat_length * np.sin(env.boat.heading)]
    p2 = [boat_x + (boat_length/2) * np.cos(env.boat.heading + 2.5), boat_y + (boat_length/2) * np.sin(env.boat.heading + 2.5)]
    p3 = [boat_x + (boat_length/2) * np.cos(env.boat.heading - 2.5), boat_y + (boat_length/2) * np.sin(env.boat.heading - 2.5)]
    
    boat_triangle = Polygon([p1, p2, p3], closed=True, color=boat_color)
    ax.add_patch(boat_triangle)

    # 5. Titolo e info
    ax.set_title(f"Step: {env.step_count} | Speed: {env.boat.speed:.1f} | Foil: {env.boat.foil}", color='white')

    # Rimuovi assi per un look più pulito
    ax.set_xticks([])
    ax.set_yticks([])

    # Converti il canvas in RGB array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    image = image[:, :, :3]
    
    return image