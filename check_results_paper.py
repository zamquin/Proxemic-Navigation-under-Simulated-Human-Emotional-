import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from social_env_paper import SocialNavEnv

def evaluate_metrics_table(env, model, num_episodes_per_emo=100):
    emotions = [-1.0, 0.0, 1.0]
    names = {-1.0: 'Negativo (Ira)', 0.0: 'Neutro', 1.0: 'Positivo (Feliz)'}
    limites_confort = {-1.0: 0.60, 0.0: 0.35, 1.0: 0.20}
    
    resultados = []
    
    for emo in emotions:
        steps_intrusion = 0
        steps_totales = 0
        success_count = 0
        distancias_frenado = []
        
        for _ in range(num_episodes_per_emo):
            obs, _ = env.reset()
            env.human_emotion = emo
            env._update_color()
            
            done = False
            truncated = False
            episode_success = False
            final_dist = env._get_distance()
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                if done and info.get("is_success", False):
                    episode_success = True
                    
                dist = info.get("dist", env._get_distance())
                final_dist = dist
                
                if dist < limites_confort[emo]:
                    steps_intrusion += 1
                steps_totales += 1
                
            if episode_success:
                success_count += 1
                distancias_frenado.append(final_dist)
                
        sii = (steps_intrusion / steps_totales) * 100 if steps_totales > 0 else 0
        success_rate = (success_count / num_episodes_per_emo) * 100
        mean_dist = np.mean(distancias_frenado) if distancias_frenado else 0.0
        std_dist = np.std(distancias_frenado) if distancias_frenado else 0.0
        
        resultados.append({
            "Emoción": names[emo],
            "SII (%)": f"{sii:.2f}%",
            "Éxito (%)": f"{success_rate:.2f}%",
            "Distancia de Frenado (m)": f"{mean_dist:.4f} ± {std_dist:.4f}"
        })
        
    df = pd.DataFrame(resultados)
    return df

def generate_thesis_style_plots(model_path, env, out_dir):
    model = PPO.load(model_path)
    
    emotions = [-1.0, 0.0, 1.0]
    names = {-1.0: 'Negativo (Ira)', 0.0: 'Neutro', 1.0: 'Positivo (Feliz)'}
    
    data = {}
    
    # Extraer datos de un episodio por emoción
    for emo in emotions:
        obs, _ = env.reset()
        env.human_emotion = emo
        env._update_color()
        name = names[emo]
        
        data[name] = {'t': [], 'dist': [], 'vel': [], 'x': [], 'y': []}
        
        done = False
        truncated = False
        step_idx = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            data[name]['t'].append(step_idx * env.dt)
            data[name]['dist'].append(info.get("dist", env._get_distance()))
            data[name]['vel'].append(info.get("v", env.v_actual))
            data[name]['x'].append(env.robot_pose[0])
            data[name]['y'].append(env.robot_pose[1])
            
            step_idx += 1
            
        for key in data[name]:
            data[name][key] = np.array(data[name][key])

    # Graficar al estilo Tesis: Grid 3x3
    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    colors = {
        'Negativo (Ira)': '#d62728',
        'Neutro': '#e6b800',
        'Positivo (Feliz)': '#2ca02c'
    }
    
    emotions_order = ['Negativo (Ira)', 'Neutro', 'Positivo (Feliz)']
    limits = {
        'Negativo (Ira)': (0.6, 0.8),
        'Neutro': (0.35, 0.6),
        'Positivo (Feliz)': (0.2, 0.35)
    }
    
    human_x, human_y = 2.0, 0.0
    
    for idx, name in enumerate(emotions_order):
        if name not in data: continue
        emo_data = data[name]
        color = colors[name]
        inner, outer = limits[name]
        
        # --- Col 1: Trayectoria X-Y con Círculos ---
        ax_traj = axs[idx, 0]
        ax_traj.plot(emo_data['x'], emo_data['y'], color=color, linewidth=2.5, label='Trayectoria Robot')
        ax_traj.plot(human_x, human_y, 'ro', markersize=8, label='Humano')
        
        # Círculos
        circle_inner = plt.Circle((human_x, human_y), inner, color=color, fill=False, linestyle='--', linewidth=1.5, label=f'Límite ({inner}m)')
        circle_outer = plt.Circle((human_x, human_y), outer, color='gray', fill=False, linestyle=':', linewidth=1.2, label=f'Meta ({outer}m)')
        ax_traj.add_patch(circle_inner)
        ax_traj.add_patch(circle_outer)
        
        ax_traj.set_xlim(0.0, 3.0)
        ax_traj.set_ylim(-1.5, 1.5)
        ax_traj.set_xlabel("X (m)", fontsize=10)
        ax_traj.set_ylabel("Y (m)", fontsize=10)
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, linestyle=':', alpha=0.6)
        ax_traj.legend(loc='upper left', fontsize=8)
        ax_traj.set_title(f"Trayectoria (Vista Superior) - {name}", fontsize=11, fontweight='bold')
        
        # --- Col 2: Distancia vs Tiempo ---
        ax_dist = axs[idx, 1]
        ax_dist.plot(emo_data['t'], emo_data['dist'], color=color, linewidth=2.5, label='Distancia Real')
        ax_dist.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
        
        ax_dist.axhline(y=inner, color=color, linestyle='--', alpha=0.5, label=f'Seguridad ({inner}m)')
        ax_dist.axhspan(inner, outer, color=color, alpha=0.05, label='Zona de Parking')
            
        ax_dist.set_xlabel("Tiempo (s)", fontsize=10)
        ax_dist.set_ylabel("Distancia $\\rho$ (m)", fontsize=10)
        ax_dist.grid(True, linestyle=':', alpha=0.6)
        ax_dist.legend(loc='upper right', fontsize=8)
        ax_dist.set_title(f"Evolución Distancia - {name}", fontsize=11, fontweight='bold')
        
        # --- Col 3: Velocidad vs Tiempo ---
        ax_vel = axs[idx, 2]
        ax_vel.plot(emo_data['t'], emo_data['vel'], color=color, linestyle='-', linewidth=2.5, label='Velocidad $v$')
        ax_vel.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        ax_vel.set_xlabel("Tiempo (s)", fontsize=10)
        ax_vel.set_ylabel("Velocidad $v$ (m/s)", fontsize=10)
        ax_vel.grid(True, linestyle=':', alpha=0.6)
        ax_vel.legend(loc='lower right', fontsize=8)
        ax_vel.set_title(f"Velocidad - {name}", fontsize=11, fontweight='bold')

    plt.suptitle("Análisis Cinemático: Trayectoria, Distancia y Velocidad", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "resultados_completos_tesis.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    models_dir = "models/PPO"
    output_dir = "graficos_diagnostico"
    SEED = 42
    
    env = SocialNavEnv(smooth_penalty=True, phase=3)
    model_path = f"{models_dir}/ppo_social_nav_seed{SEED}.zip"
    
    print("Cargando modelo...")
    model = PPO.load(model_path)
    
    print("\nGenerando tabla de métricas cuantitativas (100 episodios por emoción)...")
    df = evaluate_metrics_table(env, model, num_episodes_per_emo=100)
    
    print("\n==========================================================================================")
    print("                            TABLA DE ESTADÍSTICAS FINALES                                 ")
    print("==========================================================================================")
    print(df.to_markdown(index=False))
    print("==========================================================================================\n")
    
    print("Generando gráficos de diagnóstico al estilo de la tesis (Grid 3x3 con círculos)...")
    generate_thesis_style_plots(model_path, env, output_dir)
    print(f"Gráfico completo guardado en: {output_dir}/resultados_completos_tesis.png")
