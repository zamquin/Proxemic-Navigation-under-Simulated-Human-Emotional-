import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from stable_baselines3 import PPO

from social_env_paper import SocialNavEnv

MODEL_PATH = "models/PPO/ppo_social_nav_seed42"
OUT_DIR = "graficos tesis"

def run_simulation(mode='normal'):
    env = SocialNavEnv()
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    obs, _ = env.reset()
    
    # Mapeo al nuevo formato Paper
    if mode == 'angry': env.human_emotion = -1.0
    elif mode == 'neutral': env.human_emotion = 0.0
    elif mode == 'happy': env.human_emotion = 1.0
    
    # Inyectar emoción forzada
    if mode != 'normal':
        obs[4] = env.human_emotion

    x = [env.robot_pose[0]]
    y = [env.robot_pose[1]]
    emotions = [env.human_emotion] 
    
    for step in range(120): # Un poco más de tiempo para que aparque suave
        obs[4] = env.human_emotion
        action, _ = model.predict(obs, deterministic=True)
        
        # Modo dinámico: cambia de emoción esporádicamente
        if mode == 'normal':
            if random.random() < 0.10:
                new_emo = random.choice([-1.0, 0.0, 1.0])
                if new_emo != env.human_emotion:
                    env.human_emotion = new_emo
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        x.append(env.robot_pose[0])
        y.append(env.robot_pose[1])
        emotions.append(env.human_emotion)
        
        if terminated or truncated: 
            break
            
    return np.array(x), np.array(y), np.array(emotions), env.human_pose

def plot_subplot_english(ax, title, x, y, emotions, human_pos):
    h_x, h_y = human_pos[0], human_pos[1]

    # Zonas Proxémicas (Mismas distancias que en check_results)
    ax.add_patch(plt.Circle((h_x, h_y), 0.80, color='#ea9999', alpha=0.15)) # Peligro alto
    ax.add_patch(plt.Circle((h_x, h_y), 0.60, color='#cc0000', fill=False, ls='--', lw=1.2)) # Limite Angry
    ax.add_patch(plt.Circle((h_x, h_y), 0.60, color='#ffe599', alpha=0.25)) 
    ax.add_patch(plt.Circle((h_x, h_y), 0.35, color='#bf9000', fill=False, ls='-', lw=1.2)) # Limite Neutral
    ax.add_patch(plt.Circle((h_x, h_y), 0.35, color='#b6d7a8', alpha=0.4))
    ax.add_patch(plt.Circle((h_x, h_y), 0.20, color='#38761d', fill=False, ls=':', lw=1.5)) # Limite Happy

    # Trayectoria coloreada según emoción
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Nuevos límites para colormap [-1.0, 0.0, 1.0]
    cmap = ListedColormap(['#d62728', '#e6b800', '#2ca02c']) # Rojo, Amarillo, Verde
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=10)
    lc.set_array(emotions[:-1])
    ax.add_collection(lc)
    
    # Humano
    ax.plot(h_x, h_y, 'ko', markersize=10, zorder=15)
    
    # Marcas donde cambia la emoción
    changes = np.where(emotions[:-1] != emotions[1:])[0] + 1
    if len(changes) > 0:
        for i in changes:
            ax.plot(x[i], y[i], 'kX', markersize=8, markeredgecolor='white', zorder=20)

    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X Position (m)', fontsize=9)
    ax.set_ylabel('Y Position (m)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')
    
    dist = np.sqrt((h_x-x[-1])**2 + (h_y-y[-1])**2)
    ax.text(-0.0, -1.3, f"Final Dist.: {dist:.2f}m", fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

def generate_trajectory_plots():
    print("Generando Gráficos de Trayectoria 2D (Versión Paper)...")
    random.seed(42)
    np.random.seed(42)
    
    res_normal = run_simulation('normal')
    res_angry = run_simulation('angry')
    res_neutral = run_simulation('neutral')
    res_happy = run_simulation('happy')

    if None in [res_normal, res_angry, res_neutral, res_happy]:
        print("Error en simulaciones.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # Combined 2x2 Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plot_subplot_english(axs[0, 0], "A. Dynamic Case (Emotion changes during approach)", *res_normal)
    plot_subplot_english(axs[0, 1], "B. Static Case: ALWAYS Negative (Angry)", *res_angry)
    plot_subplot_english(axs[1, 0], "C. Static Case: ALWAYS Neutral", *res_neutral)
    plot_subplot_english(axs[1, 1], "D. Static Case: ALWAYS Positive (Happy)", *res_happy)
    
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=4, label='Path (Positive)'),
        Line2D([0], [0], color='#e6b800', lw=4, label='Path (Neutral)'),
        Line2D([0], [0], color='#d62728', lw=4, label='Path (Negative)'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', label='Human'),
        Line2D([0], [0], color='#cc0000', linestyle='--', label='Boundary (0.6m)'),
        Line2D([0], [0], color='#bf9000', linestyle='-', label='Boundary (0.35m)'),
        Line2D([0], [0], color='#38761d', linestyle=':', label='Boundary (0.2m)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=11)
    plt.suptitle("Proxemic Behavior Comparison in 2D Space", fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    png_path = os.path.join(OUT_DIR, "thesis_plot_premium_paper.png")
    pdf_path = os.path.join(OUT_DIR, "thesis_plot_premium_paper.pdf")
    
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ¡Gráfico majestuoso guardado en {png_path} y .pdf!")

if __name__ == "__main__":
    generate_trajectory_plots()
