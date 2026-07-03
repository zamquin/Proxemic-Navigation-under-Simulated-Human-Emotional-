import os
import cv2
import numpy as np
import time
from stable_baselines3 import PPO
from deepface import DeepFace

# Suprimir logs basuras de librerías subyacentes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from webots_env_paper import WebotsSocialEnv

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/PPO/ppo_social_nav_seed42"

def main():
    print("Iniciando Webots Interactivo (Versión Paper con Visión)...")
    try:
        env = WebotsSocialEnv()
    except Exception as e:
        print(f"Error al conectar con Webots: {e}")
        return
        
    print(f"Cargando política PPO: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    # Habilitar cámara explícitamente por precaución
    camera = env.robot.getDevice("camera")
    if camera:
        camera.enable(env.timestep)
    else:
        print("Advertencia: No se detectó cámara en el robot.")
        
    obs, _ = env.reset()
    
    frame_count = 0
    ultima_emocion_str = "Buscando Rostro..."
    current_emotion_val = 0.0  # Neutro por defecto
    
    print("\n=======================================================")
    print(" INICIANDO BUCLE INFINITO (PRUEBA DE FUEGO)            ")
    print("=======================================================")
    print("1. Arranca tu simulador y asegúrate que no esté en pausa.")
    print("2. Puedes hacer click sobre el humano y moverlo con SHIFT.")
    print("3. La ventana de OpenCV procesará su emoción en tiempo real.")
    print("4. Para salir, presiona 'q' sobre la ventana de video.\n")
    
    while True:
        frame_count += 1
        
        # 1. Extraer imagen de la cámara Webots
        raw_img = camera.getImage() if camera else None
        
        if raw_img:
            # Procesar raw bytes a imagen numpy
            img = np.frombuffer(raw_img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            
            # Quitar canal Alpha y clonar para poder dibujar encima sin fallos de memoria
            img_bgr = img[:, :, :3].copy()
            
            # Analizar cada 10 frames (~3 FPS en visión) para no colapsar la CPU y permitir a Webots avanzar
            if frame_count % 10 == 0:
                try:
                    objs = DeepFace.analyze(
                        img_path = img_bgr, 
                        actions = ['emotion'], 
                        enforce_detection=False,
                        detector_backend='opencv', 
                        silent=True
                    )
                    
                    if objs and isinstance(objs, list):
                        result = objs[0]
                        region = result['region']
                        
                        # Filtro simple para ignorar detecciones fantasma pequeñas
                        if region['w'] > 20 and region['h'] > 20:
                            dom_emo = result['dominant_emotion']
                            
                            # Mapear a los valores normalizados que aprendió el agente (-1.0, 0.0, 1.0)
                            if dom_emo == 'happy':
                                current_emotion_val = 1.0
                                ultima_emocion_str = "FELIZ :)"
                            elif dom_emo == 'angry':
                                current_emotion_val = -1.0
                                ultima_emocion_str = "ENOJADO >:("
                            else:
                                current_emotion_val = 0.0
                                ultima_emocion_str = f"NEUTRO ({dom_emo})"
                        else:
                            ultima_emocion_str = "Muy lejos..."
                            
                except Exception:
                    pass
            
            # Dibujar Estado en Pantalla
            color_texto = (0, 255, 0) if current_emotion_val > 0 else ((0, 0, 255) if current_emotion_val < 0 else (0, 255, 255))
            cv2.putText(img_bgr, f"Emocion: {ultima_emocion_str}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 2)
            cv2.imshow("Robot Z-UP Visión y Control", img_bgr)
            
            # Salir presionando q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nSaliendo del bucle interactivo...")
                break

        # 2. Actualizar el estado con lo que la cámara ve
        obs[4] = current_emotion_val
        env.human_emotion = current_emotion_val
        
        # 3. Cerebro (PPO) decide la acción
        action, _ = model.predict(obs, deterministic=True)
        
        # 4. Cuerpo (E-Puck) ejecuta los motores
        obs, reward, done, truncated, info = env.step(action)
        
        # Feedback por consola (aprox cada segundo asumiendo 30 iteraciones)
        if frame_count % 30 == 0:
            dist_real = obs[0] * 10.0
            v_cmd = info.get('v', 0.0)
            print(f"Distancia: {dist_real:.3f}m | Emoción: {ultima_emocion_str} | Vel: {v_cmd:.3f}m/s")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
