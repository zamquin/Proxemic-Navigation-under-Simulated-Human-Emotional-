import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Importar el nuevo entorno de Webots con físicas del Paper
from webots_env_paper import WebotsSocialEnv

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/PPO/ppo_social_nav_seed42"
CSV_PATH = "resultados_webots_paper.csv"

def run_experiments():
    print("Iniciando conexión con Webots (Versión Paper)...")
    try:
        env = WebotsSocialEnv()
    except Exception as e:
        print(f"Error al iniciar el entorno de Webots: {e}")
        print("Asegúrate de tener Webots abierto con el mundo cargado y en pausa antes de ejecutar.")
        return
        
    print(f"Cargando cerebro entrenado desde: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Emociones a testear (Mapeadas a -1.0, 0.0, 1.0 como en la nueva política)
    emotions = {
        'Negativo (Ira)': -1.0,
        'Neutro': 0.0,
        'Positivo (Feliz)': 1.0
    }
    
    # Historial para guardar en CSV
    all_data = []
    
    # Estructura del plot temporal
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex='col')
    colors = {-1.0: '#d62728', 0.0: '#e6b800', 1.0: '#2ca02c'}
    
    for idx, (name, emo_val) in enumerate(emotions.items()):
        print(f"\n>>> Corriendo experimento para Humano {name} ({emo_val}) en Webots...")
        
        obs, _ = env.reset()
        env.human_emotion = emo_val
        obs[4] = emo_val
        
        times = []
        distances = []
        velocities = []
        
        step_counter = 0
        timestep_sec = env.timestep / 1000.0 # Convertir ms a segundos
        
        done = False
        while not done and step_counter < 500: # Límite
            # El cerebro predice la acción normalizada basada en Webots
            action, _ = model.predict(obs, deterministic=True)
            
            # Ejecutar acción en Webots
            obs, reward, done, truncated, info = env.step(action)
            
            # Extraer variables físicas reales de Webots
            dist = obs[0] * 10.0 # Des-normalizar la distancia solo para graficar
            v_cmd = info.get('v', 0.0) # Obtener la velocidad real des-normalizada calculada en step()
            t = step_counter * timestep_sec
            
            times.append(t)
            distances.append(dist)
            velocities.append(v_cmd)
            
            # Guardar para el CSV
            all_data.append({
                'Emocion': name,
                'Tiempo': t,
                'Distancia': dist,
                'Velocidad_Comandada': v_cmd
            })
            
            # Imprimir en consola
            print(f"  Paso {step_counter:03d} | t={t:.2f}s | Dist={dist:.3f}m | Vel={v_cmd:.3f}m/s")
            
            step_counter += 1
            
        print(f"  Experimento completado en {step_counter} pasos.")
        
        # Graficar en tiempo real en la figura
        color = colors[emo_val]
        
        # Columna 1: Distancia vs Tiempo
        axs[idx, 0].plot(times, distances, color=color, linewidth=2.5, label='Distancia Real (Webots)')
        axs[idx, 0].axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
        
        # Límites proxémicos
        if emo_val == -1.0:
            axs[idx, 0].axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Límite Seguridad (0.6m)')
            axs[idx, 0].axhspan(0.6, 0.8, color='red', alpha=0.05, label='Zona de Parking')
        elif emo_val == 0.0:
            axs[idx, 0].axhline(y=0.35, color='orange', linestyle='--', alpha=0.5, label='Límite Seguridad (0.35m)')
            axs[idx, 0].axhspan(0.35, 0.6, color='orange', alpha=0.05, label='Zona de Parking')
        elif emo_val == 1.0:
            axs[idx, 0].axhline(y=0.20, color='green', linestyle='--', alpha=0.5, label='Límite Seguridad (0.2m)')
            axs[idx, 0].axhspan(0.2, 0.35, color='green', alpha=0.05, label='Zona de Parking')
            
        axs[idx, 0].set_ylabel("Distancia $\\rho$ (m)", fontsize=10)
        axs[idx, 0].grid(True, linestyle=':', alpha=0.6)
        axs[idx, 0].legend(loc='upper right', fontsize=8)
        axs[idx, 0].set_title(f"Webots: Evolución Distancia - {name}", fontsize=10, fontweight='bold')
        
        # Columna 2: Velocidad vs Tiempo
        axs[idx, 1].plot(times, velocities, color=color, linestyle='-', linewidth=2.5, label='Velocidad Comandada $v$')
        axs[idx, 1].axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        axs[idx, 1].set_ylabel("Velocidad $v$ (m/s)", fontsize=10)
        axs[idx, 1].grid(True, linestyle=':', alpha=0.6)
        axs[idx, 1].legend(loc='lower right', fontsize=8)
        axs[idx, 1].set_title(f"Webots: Velocidad vs Tiempo - {name}", fontsize=10, fontweight='bold')

    axs[2, 0].set_xlabel("Tiempo (segundos)", fontsize=11)
    axs[2, 1].set_xlabel("Tiempo (segundos)", fontsize=11)
    
    plt.suptitle("Experimento Real en Webots: Política Físicamente Honesta", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Guardar gráfico
    out_dir = "graficos tesis"
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "resultados_webots_paper.png")
    pdf_path = os.path.join(out_dir, "resultados_webots_paper.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    # Guardar en CSV
    print(f"\nGuardando datos en CSV: {CSV_PATH}")
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Emocion', 'Tiempo', 'Distancia', 'Velocidad_Comandada'])
        writer.writeheader()
        writer.writerows(writer_rows := all_data)
        
    print("¡Experimentos en Webots finalizados!")
    print(f"Gráficos guardados en: {png_path} y {pdf_path}")

if __name__ == "__main__":
    run_experiments()
