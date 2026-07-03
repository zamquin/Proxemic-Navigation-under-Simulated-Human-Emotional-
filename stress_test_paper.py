import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from stable_baselines3 import PPO
from social_env_paper import SocialNavEnv

# Configuración
MODEL_PATH = "models/PPO/ppo_social_nav_seed42"
OUT_DIR = "graficos tesis"

def run_trap_scenario():
    env = SocialNavEnv()
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Modelo no encontrado: {e}")
        return None

    # 1. Forzamos inicio FELIZ (1.0)
    obs, _ = env.reset()
    env.human_emotion = 1.0 # Feliz
    
    # Hack: Desactivamos la actualización aleatoria de emoción del entorno (si la tuviese)
    x, y, emotions = [env.robot_pose[0]], [env.robot_pose[1]], [env.human_emotion]
    
    trap_triggered = False
    print("Iniciando acercamiento (Humano Feliz [1.0])...")

    # Aumentamos límite de pasos para que alcance a escapar
    for i in range(250):
        # Asegurar que le pasamos la emoción que queremos
        obs[4] = env.human_emotion
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        
        # Calcular distancia real al humano
        dx = env.human_pose[0] - env.robot_pose[0]
        dy = env.human_pose[1] - env.robot_pose[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # 2. LA TRAMPA: Cuando esté muy cerca (< 0.35m), ¡Cambio a NEGATIVO (-1.0)!
        if not trap_triggered and dist < 0.35:
            print(f"¡TRAMPA ACTIVADA en paso {i}! Distancia: {dist:.2f}m -> Cambio a Ira (-1.0)")
            env.human_emotion = -1.0 # Ira
            # Actualizamos la observación manualmente para que el cerebro se entere YA
            obs[4] = -1.0 
            trap_triggered = True
            
        x.append(env.robot_pose[0])
        y.append(env.robot_pose[1])
        emotions.append(env.human_emotion)
        
        # Evitar que 'done' detenga la simulación prematuramente por violar la regla de Ira 
        # (queremos ver si logra corregirlo)
        if truncated: 
            break
            
    return np.array(x), np.array(y), np.array(emotions), env.human_pose

def plot_trap(x, y, emotions, human_pos):
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Trayectoria Multicolor
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Colormap Paper: -1.0 (Rojo), 0.0 (Amarillo), 1.0 (Verde)
    cmap = ListedColormap(['#d62728', '#e6b800', '#2ca02c']) 
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
    lc.set_array(emotions[:-1])
    ax.add_collection(lc)
    
    h_x, h_y = human_pos[0], human_pos[1]
    
    # Humano
    ax.plot(h_x, h_y, 'o', color='black', markersize=15, label='Humano', zorder=20)
    
    # Zonas CLAVE
    # Zona Feliz (Donde entra confiado)
    c_pos = plt.Circle((h_x, h_y), 0.20, color='#2ca02c', fill=False, linestyle=':', linewidth=2, label='Límite Feliz (0.2m)')
    c_pos_fill = plt.Circle((h_x, h_y), 0.20, color='#b6d7a8', alpha=0.3)
    # Zona Ira (Límite del que debe escapar)
    c_neg = plt.Circle((h_x, h_y), 0.60, color='#d62728', fill=False, linestyle='--', linewidth=2, label='Límite Ira (0.6m)')
    c_neg_fill = plt.Circle((h_x, h_y), 0.60, color='#ea9999', alpha=0.1)
    
    ax.add_patch(c_pos_fill)
    ax.add_patch(c_neg_fill)
    ax.add_patch(c_pos)
    ax.add_patch(c_neg)

    # Marcar el momento exacto de la trampa
    changes = np.where(emotions[:-1] != emotions[1:])[0] + 1
    for i in changes:
        ax.plot(x[i], y[i], 'X', color='black', markersize=12, zorder=25, label='Momento de la Trampa')
        ax.annotate("¡TRAMPA\nACTIVADA!", (x[i], y[i]), xytext=(-30, 20), textcoords='offset points', 
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    # Márgenes centrados en el humano
    margin = 1.2
    ax.set_xlim(h_x - margin, h_x + margin)
    ax.set_ylim(h_y - margin, h_y + margin)
    
    ax.set_title("Test de Estrés Dinámico: Reacción de Escape (Feliz -> Ira)", fontsize=13, fontweight='bold')
    ax.set_xlabel('Posición X (m)')
    ax.set_ylabel('Posición Y (m)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "stress_test_paper.png")
    pdf_path = os.path.join(OUT_DIR, "stress_test_paper.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de Stress Test guardado en: {png_path}")

if __name__ == "__main__":
    x, y, emo, h_pos = run_trap_scenario()
    if x is not None:
        plot_trap(x, y, emo, h_pos)
