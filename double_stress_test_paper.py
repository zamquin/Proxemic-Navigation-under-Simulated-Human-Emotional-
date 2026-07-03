import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from stable_baselines3 import PPO

from social_env_paper import SocialNavEnv

MODEL_PATH = "models/PPO/ppo_social_nav_seed42"
OUT_DIR = "graficos tesis"

def run_double_trap_scenario():
    env = SocialNavEnv()
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    obs, _ = env.reset()
    
    # Forzar posición inicial más cerca pero con curva
    env.robot_pose = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    env.human_pose = np.array([2.0, 0.0], dtype=np.float32)
    
    # Recalcular observación inicial (Hack directo para este script)
    dx = env.human_pose[0] - env.robot_pose[0]
    dy = env.human_pose[1] - env.robot_pose[1]
    rho = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx) - env.robot_pose[2]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    
    # 1. Empieza Feliz (+1.0)
    env.human_emotion = 1.0 
    obs = np.array([rho / 10.0, alpha / np.pi, 0.0, 0.0, env.human_emotion], dtype=np.float32)
    
    x, y, emotions = [env.robot_pose[0]], [env.robot_pose[1]], [env.human_emotion]
    velocities = [0.0] # Inicia en reposo
    times = [0.0]
    
    dt = 0.1 # Timestep del entorno
    
    print("Starting approach (Happy [1.0])...")

    trap_1_triggered = False
    trap_2_triggered = False
    
    for step in range(500):
        obs[4] = env.human_emotion
        action, _ = model.predict(obs, deterministic=True)
        
        # Calcular distancia real al humano
        dx = env.human_pose[0] - env.robot_pose[0]
        dy = env.human_pose[1] - env.robot_pose[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # EL GUIÓN DE LA MONTAÑA RUSA BASADO EN TIEMPO (Dejar que estacione primero)
        # Trampa 1: Paso 180 (18 segundos). Ya debería estar estacionado cómodamente en la zona verde.
        if step == 180:
            env.human_emotion = -1.0
            obs[4] = -1.0
            trap_1_triggered = True
            print(f"Step {step} (Dist={dist:.2f}m): Changed to Negative (Angry)!")
            
        # Trampa 2: Paso 330 (33 segundos). Ya debería haber retrocedido y estacionado lejos.
        if step == 330:
            env.human_emotion = 1.0
            obs[4] = 1.0
            trap_2_triggered = True
            # ¡PERTURBACIÓN FÍSICA! Giramos el robot 90 grados para forzar una curva de retorno
            env.robot_pose[2] += np.pi / 2.0
            print(f"Step {step} (Dist={dist:.2f}m): Changed to Positive (Happy) + Heading Perturbation!")

        obs, _, done, truncated, info = env.step(action)
        
        x.append(env.robot_pose[0])
        y.append(env.robot_pose[1])
        emotions.append(env.human_emotion)
        
        # Recuperamos la velocidad LINEAL REAL calculada por el entorno
        v = info.get('v', 0.0)
        velocities.append(v)
        times.append((step + 1) * dt)
        
        # Eliminamos el break para forzar los 500 pasos completos ignorando el fin del episodio
        # if done or truncated: 
        #    break
            
    return np.array(x), np.array(y), np.array(emotions), np.array(velocities), np.array(times), env.human_pose

def plot_double_stress_and_kinematics(x, y, emo, vels, times, h_pos):
    # Cálculo Cinemático
    dt = 0.1
    # Aceleración = dv / dt
    accels = np.zeros_like(vels)
    accels[1:] = (vels[1:] - vels[:-1]) / dt
    
    # Jerk = da / dt
    jerks = np.zeros_like(accels)
    jerks[1:] = (accels[1:] - accels[:-1]) / dt
    
    # Setup de la Figura
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], wspace=0.25, hspace=0.4)
    
    # ==========================
    # PANEL IZQUIERDO: TRAYECTORIA 2D
    # ==========================
    ax_traj = fig.add_subplot(gs[:, 0])
    h_x, h_y = h_pos[0], h_pos[1]
    
    # Donas Proxémicas
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.80, color='#ea9999', alpha=0.15))
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.60, color='#cc0000', fill=False, ls='--', lw=1.2))
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.60, color='#ffe599', alpha=0.25))
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.35, color='#bf9000', fill=False, ls='-', lw=1.2))
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.35, color='#b6d7a8', alpha=0.4))
    ax_traj.add_patch(plt.Circle((h_x, h_y), 0.20, color='#38761d', fill=False, ls=':', lw=1.5))

    # Trayectoria coloreada
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(['#d62728', '#e6b800', '#2ca02c']) 
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.0, zorder=10)
    lc.set_array(emo[:-1])
    ax_traj.add_collection(lc)
    
    # Marcadores de Eventos (Engaños)
    changes = np.where(emo[:-1] != emo[1:])[0] + 1
    for i, idx in enumerate(changes):
        new_emo = emo[idx]
        label = "Angry!" if new_emo < 0 else "Happy!"
        y_offset = 25 if i % 2 == 0 else -25
        ax_traj.plot(x[idx], y[idx], 'kX', markersize=10, markeredgecolor='white', zorder=20)
        ax_traj.annotate(label, (x[idx], y[idx]), xytext=(0, y_offset), textcoords='offset points', 
                         ha='center', fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.9))

    ax_traj.plot(h_x, h_y, 'ko', markersize=12, zorder=15, label='Human')
    ax_traj.set_xlim(-0.8, 2.5); ax_traj.set_ylim(-1.2, 1.2)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_title("A. 2D Path: Double Stress Test (Happy -> Angry -> Happy)", fontweight='bold')
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    ax_traj.set_aspect('equal')
    
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=3, label='Positive Context'),
        Line2D([0], [0], color='#d62728', lw=3, label='Negative Context'),
        Line2D([0], [0], color='black', marker='X', lw=0, label='Emotion Switch'),
        Line2D([0], [0], color='#cc0000', ls='--', label='Negative Boundary (0.6m)')
    ]
    ax_traj.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # ==========================
    # PANEL DERECHO: CINEMÁTICA Y JERK
    # ==========================
    change_times = [times[idx] for idx in changes]
    
    # Velocidad
    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.plot(times, vels, color='#1f77b4', lw=2)
    ax_v.set_ylabel('Velocity\n(m/s)', fontweight='bold')
    ax_v.set_title("B. Kinematic Profiling: Jerk Analysis", fontweight='bold')
    ax_v.grid(True, alpha=0.4)
    
    # Aceleración
    ax_a = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_a.plot(times, accels, color='#ff7f0e', lw=2)
    ax_a.set_ylabel('Acceleration\n(m/s²)', fontweight='bold')
    ax_a.grid(True, alpha=0.4)
    
    # Jerk
    ax_j = fig.add_subplot(gs[2, 1], sharex=ax_v)
    ax_j.plot(times, jerks, color='#9467bd', lw=2)
    ax_j.set_ylabel('Jerk\n(m/s³)', fontweight='bold')
    ax_j.set_xlabel('Time (s)', fontweight='bold')
    ax_j.grid(True, alpha=0.4)
    
    # Lineas verticales en los momentos de engaño
    for ax in [ax_v, ax_a, ax_j]:
        for t in change_times:
            ax.axvline(x=t, color='black', linestyle='--', alpha=0.5)
            
    # Marcar el Jerk Absoluto Máximo (Tirón pico)
    max_jerk = np.max(np.abs(jerks))
    ax_j.text(0.05, 0.85, f"Max Abs Jerk: {max_jerk:.2f} m/s³", transform=ax_j.transAxes, 
              fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#9467bd'))

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "thesis_double_stress_test_paper.png")
    pdf_path = os.path.join(OUT_DIR, "thesis_double_stress_test_paper.pdf")
    
    # Ocultar etiquetas X de V y A para que quede limpio
    plt.setp(ax_v.get_xticklabels(), visible=False)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    
    fig.align_ylabels([ax_v, ax_a, ax_j])
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Double Stress Test & Jerk Plot saved to: {png_path}")

if __name__ == "__main__":
    res = run_double_trap_scenario()
    if res is not None:
        plot_double_stress_and_kinematics(*res)
