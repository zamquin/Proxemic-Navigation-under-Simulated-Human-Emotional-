import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Límites físicos del e-puck
MAX_V = 0.12
MIN_V = -0.06
MAX_W = 1.0

def generate_data(model):
    """Realiza un barrido de rho para las 3 emociones con alpha=0.0 y velocidades iniciales en 0."""
    rho_grid = np.linspace(0.05, 3.0, 500)
    data = {
        'angry': {'rho': [], 'v': [], 'w': []},
        'neutral': {'rho': [], 'v': [], 'w': []},
        'happy': {'rho': [], 'v': [], 'w': []}
    }
    
    # Emociones codificadas como en el nuevo entorno
    emotions = {
        'angry': -1.0,
        'neutral': 0.0,
        'happy': 1.0
    }
    
    for key, emo_val in emotions.items():
        for rho in rho_grid:
            # Observación normalizada: [rho/10, alpha/pi, v_actual/max_v, w_actual/max_w, emotion]
            obs = np.array([rho / 10.0, 0.0, 0.0, 0.0, emo_val], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            
            # Reescalado idéntico al de step()
            raw_v = float(np.clip(action[0], -1.0, 1.0))
            raw_w = float(np.clip(action[1], -1.0, 1.0))
            
            v = MIN_V + (raw_v + 1.0) * 0.5 * (MAX_V - MIN_V)
            w = raw_w * MAX_W
            
            v_cmd = float(np.clip(v, MIN_V, MAX_V))
            w_cmd = float(np.clip(w, -MAX_W, MAX_W))
            
            data[key]['rho'].append(rho)
            data[key]['v'].append(v_cmd)
            data[key]['w'].append(w_cmd)
            
        data[key]['rho'] = np.array(data[key]['rho'])
        data[key]['v'] = np.array(data[key]['v'])
        data[key]['w'] = np.array(data[key]['w'])
        
    return data

def plot_control_law_en(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Parking zones (very soft shading)
    ax.axvspan(0.60, 0.80, color='#cc0000', alpha=0.08, label='Parking Zone (Angry)')
    ax.axvspan(0.35, 0.60, color='#bf9000', alpha=0.08, label='Parking Zone (Neutral)')
    ax.axvspan(0.20, 0.35, color='#38761d', alpha=0.08, label='Parking Zone (Happy)')
    
    # Vertical lines for proxemic safety boundaries (d_safe)
    ax.axvline(x=0.60, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=0.35, color='#e6b800', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=0.20, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.7)
    
    # Annotation labels (staggered vertically to prevent overlapping)
    ax.text(0.58, -0.05, 'Angry Boundary\n(0.6m)', color='#cc0000', fontsize=8, fontweight='bold')
    ax.text(0.33, -0.04, 'Neutral Boundary\n(0.35m)', color='#bf9000', fontsize=8, fontweight='bold')
    ax.text(0.18, -0.03, 'Happy Boundary\n(0.2m)', color='#38761d', fontsize=8, fontweight='bold')
    
    # Plot velocity curves
    ax.plot(data['angry']['rho'], data['angry']['v'], color='#d62728', linewidth=2.5, label='Policy: Negative (Angry)')
    ax.plot(data['neutral']['rho'], data['neutral']['v'], color='#e6b800', linewidth=2.5, label='Policy: Neutral')
    ax.plot(data['happy']['rho'], data['happy']['v'], color='#2ca02c', linewidth=2.5, label='Policy: Positive (Happy)')
    
    # Reference zero velocity line
    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    
    # Decoration
    ax.set_title("Learned Control Law: Commanded Linear Velocity $v$ vs Distance $\\rho$", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Distance to Human $\\rho$ (m)", fontsize=11)
    ax.set_ylabel("Commanded Linear Velocity $v$ (m/s)", fontsize=11)
    
    # Ajustar ejes para el rango real del e-puck (-0.06 a 0.12)
    ax.set_xlim(1.5, 0.0)
    ax.set_ylim(-0.08, 0.14)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 0, 4, 1, 5, 2] # Curves first, then zones
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=9, framealpha=0.95)
    
    # Save
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "control_law_velocity_paper_en.png")
    pdf_path = os.path.join(out_dir, "control_law_velocity_paper_en.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico Inglés guardado: {png_path} y {pdf_path}")
    return png_path

def main():
    MODEL_PATH = "models/PPO/ppo_social_nav_seed42.zip"
    OUT_DIR = "graficos tesis"
    
    print("Cargando modelo PPO de la versión Paper...")
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error cargando el modelo en {MODEL_PATH}: {e}")
        return
        
    print("Extrayendo datos de la ley de control...")
    data = generate_data(model)
    
    print("Generando gráficos en inglés...")
    plot_control_law_en(data, OUT_DIR)
    
    print("Proceso completado exitosamente!")

if __name__ == "__main__":
    main()
