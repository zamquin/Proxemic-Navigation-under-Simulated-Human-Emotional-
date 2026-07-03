import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from stable_baselines3 import PPO

# Asegurarse de importar el entorno Paper
from social_env_paper import SocialNavEnv

MODEL_PATH = "models/PPO/ppo_social_nav_seed42"
OUT_DIR = "graficos tesis"

def simulate_random_spawn(env, model, emotion=0.0):
    """
    Spawnea al robot en una coordenada aleatoria en un rango extendido (6x6m),
    dejando que alcance la meta.
    """
    obs, _ = env.reset()
    
    # Campo de spawn extendido: [-3.0, 3.0]
    rand_x = np.random.uniform(-3.0, 3.0)
    rand_y = np.random.uniform(-3.0, 3.0)
    rand_theta = np.random.uniform(-np.pi, np.pi)
    
    env.robot_pose = np.array([rand_x, rand_y, rand_theta], dtype=np.float32)
    env.human_emotion = emotion
    
    # Recalcular observación inicial para el PPO normalizado
    dx = env.human_pose[0] - env.robot_pose[0]
    dy = env.human_pose[1] - env.robot_pose[1]
    rho = np.sqrt(dx**2 + dy**2)
    
    # Heading y Alpha
    alpha = np.arctan2(dy, dx) - env.robot_pose[2]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    
    # Obs: rho/10, alpha/pi, v/max_v, w/max_w, emotion
    obs = np.array([rho / 10.0, alpha / np.pi, 0.0, 0.0, env.human_emotion], dtype=np.float32)
    
    x_hist = [env.robot_pose[0]]
    y_hist = [env.robot_pose[1]]
    emotions = [env.human_emotion]
    
    # Aumentamos el límite a 600 para permitirle viajar desde tan lejos (0.12 m/s requiere paciencia)
    for _ in range(600):
        obs[4] = env.human_emotion
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        
        x_hist.append(env.robot_pose[0])
        y_hist.append(env.robot_pose[1])
        emotions.append(env.human_emotion)
        
        if done or truncated: 
            break
            
    return np.array(x_hist), np.array(y_hist), np.array(emotions), env.human_pose

def plot_generalization():
    print("Iniciando Test 360 Extendido (Generalización Espacial)...")
    env = SocialNavEnv()
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Obtenemos la posición del humano directamente del entorno para evitar offsets gráficos
    h_x, h_y = env.human_pose[0], env.human_pose[1]
    
    # Zonas Proxémicas (Mismas distancias del check_results)
    ax.add_patch(plt.Circle((h_x, h_y), 0.80, color='#ea9999', alpha=0.15))
    ax.add_patch(plt.Circle((h_x, h_y), 0.60, color='#cc0000', fill=False, ls='--', lw=1.0))
    ax.add_patch(plt.Circle((h_x, h_y), 0.60, color='#ffe599', alpha=0.25))
    ax.add_patch(plt.Circle((h_x, h_y), 0.35, color='#bf9000', fill=False, ls='-', lw=1.0))
    ax.add_patch(plt.Circle((h_x, h_y), 0.35, color='#b6d7a8', alpha=0.4))
    ax.add_patch(plt.Circle((h_x, h_y), 0.20, color='#38761d', fill=False, ls=':', lw=1.0))

    # Simularemos 16 trayectorias independientes en total
    # Mapeo Paper: -1.0 (Ira), 0.0 (Neutro), 1.0 (Feliz)
    scenarios = [(0.0, 8), (-1.0, 4), (1.0, 4)]
    
    np.random.seed(42) # Estabilidad en los renders visuales
    
    print("Simulando trayectorias desde márgenes extremos (6x6 metros)...")
    for emotion, count in scenarios:
        for i in range(count):
            x, y, emo, human_pos = simulate_random_spawn(env, model, emotion)
            
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Nuevos límites del colormap
            cmap = ListedColormap(['#d62728', '#e6b800', '#2ca02c']) 
            norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
            
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, zorder=10, alpha=0.8)
            lc.set_array(emo[:-1])
            ax.add_collection(lc)
            
            # Dibujar un punto que indique desde DÓNDE partió
            color_start = '#e6b800' if emotion == 0.0 else ('#d62728' if emotion == -1.0 else '#2ca02c')
            
            # Flechas indicando la rotación inicial con la que apareció el robot
            dx = x[1]-x[0] if len(x)>1 else 0.1
            dy = y[1]-y[0] if len(y)>1 else 0.0
            norm_factor = np.sqrt(dx**2 + dy**2) + 1e-6
            dx, dy = dx/norm_factor*0.1, dy/norm_factor*0.1
            
            ax.plot(x[0], y[0], 'o', color=color_start, markersize=5, zorder=12)
            ax.annotate('', xy=(x[0]+dx*3, y[0]+dy*3), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="-|>", color='black', lw=1.0), zorder=13)

    # Humano
    ax.plot(h_x, h_y, 'ko', markersize=10, zorder=15)
    
    # Campo visual expandido para que quepa todo el test
    margin = 3.5
    ax.set_xlim(h_x - margin, h_x + margin)
    ax.set_ylim(h_y - margin, h_y + margin)
    
    ax.set_xlabel('Posición X (m)', fontsize=11)
    ax.set_ylabel('Posición Y (m)', fontsize=11)
    ax.set_title("Test de Estrés Espacial 360° (Campo Ampliado a 6x6 metros)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "test_360_extendido_paper.png")
    pdf_path = os.path.join(OUT_DIR, "test_360_extendido_paper.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de Estrés 360 guardado con éxito en: {png_path}")

if __name__ == "__main__":
    plot_generalization()
