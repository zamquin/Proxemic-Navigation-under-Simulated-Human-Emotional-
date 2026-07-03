import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import sys

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Intenta encontrar Webots automáticamente
WEBOTS_HOME = r"C:\Program Files\Webots"
if not os.path.exists(WEBOTS_HOME):
    WEBOTS_HOME = os.path.expanduser(r"~\AppData\Local\Programs\Webots")

# Cargar librerías
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))

# Configurar DLLs para Windows
if sys.platform == 'win32':
    controller_path = os.path.join(WEBOTS_HOME, 'lib', 'controller')
    os.environ['PATH'] = controller_path + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(controller_path)
        except: pass

try:
    from controller import Supervisor
except ImportError:
    print("¡Error Crítico! No se pudo importar el controlador de Webots.")
    sys.exit(1)

class WebotsSocialEnv(gym.Env):
    """
    Entorno Webots Z-UP con Física Real y Lógica Social (Versión Paper).
    """
    def __init__(self):
        super().__init__()
        
        # Conexión
        try:
            self.robot = Supervisor()
        except:
            print("Error: Webots no está corriendo. Dale al Play.")
            sys.exit(1)
            
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.robot_node = self.robot.getFromDef("E_PUCK")
        self.human_node = self.robot.getFromDef("HUMAN")
        
        if self.robot_node is None or self.human_node is None:
            print("ERROR: Faltan los DEFs 'E_PUCK' o 'HUMAN' en Webots.")
            sys.exit(1)
        
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Cinemática E-puck
        self.axle_length = 0.052 
        self.wheel_radius = 0.0205 
        self.max_motor_speed = 6.20 
        
        # Límites del entorno simulado (Paper)
        self.max_v = 0.12 
        self.min_v = -0.06 
        self.max_w = 1.0 

        # Acción normalizada [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observación normalizada: [rho/10, alpha/pi, v/max_v, w/max_w, emotion]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.human_emotion = 0.0
        self.violation_time = 0
        
        self.v_actual = 0.0
        self.w_actual = 0.0

    def step(self, action):
        # Desnormalización de Acción
        raw_v = float(np.clip(action[0], -1.0, 1.0))
        raw_w = float(np.clip(action[1], -1.0, 1.0))
        
        v_linear = self.min_v + (raw_v + 1.0) * 0.5 * (self.max_v - self.min_v)
        v_angular = raw_w * self.max_w
        
        v_linear = float(np.clip(v_linear, self.min_v, self.max_v))
        v_angular = float(np.clip(v_angular, -self.max_w, self.max_w))
        
        self.v_actual = v_linear
        self.w_actual = v_angular
        
        # Cinemática Diferencial
        left_speed = (v_linear - v_angular * self.axle_length / 2.0) / self.wheel_radius
        right_speed = (v_linear + v_angular * self.axle_length / 2.0) / self.wheel_radius
        
        left_speed = np.clip(left_speed, -self.max_motor_speed, self.max_motor_speed)
        right_speed = np.clip(right_speed, -self.max_motor_speed, self.max_motor_speed)
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            # self.camera.recognitionEnable(self.timestep) # <- Eliminado para evitar errores de OpenCV

        # Paso de simulación
        self.robot.step(self.timestep)
        
        # Cinemática 
        r_pos = self.robot_node.getPosition()
        h_pos = self.human_node.getPosition()
        r_rot = self.robot_node.getOrientation()
        
        dx = h_pos[0] - r_pos[0]
        dy = h_pos[1] - r_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        heading = np.arctan2(r_rot[3], r_rot[0])
        angle_to_target = np.arctan2(dy, dx)
        alpha = angle_to_target - heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        
        reward = 0.0 
        terminated = False
        if dist < 0.12: terminated = True
            
        # Observación Normalizada
        obs = np.array([
            np.clip(dist / 10.0, 0.0, 1.0), 
            alpha / np.pi, 
            np.clip(self.v_actual / self.max_v, -1.0, 1.0), 
            np.clip(self.w_actual / self.max_w, -1.0, 1.0), 
            self.human_emotion
        ], dtype=np.float32)
        
        info = {'v': self.v_actual}
        
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.v_actual = 0.0
        self.w_actual = 0.0
        
        self.robot_node.getField("translation").setSFVec3f([0.0, 0.0, 0.0])
        self.robot_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
        
        self.robot_node.resetPhysics()
        self.robot.step(self.timestep)
        
        # Emociones Paper
        self.human_emotion = random.choice([-1.0, 0.0, 1.0])
        self.violation_time = 0
        
        r_pos = self.robot_node.getPosition()
        h_pos = self.human_node.getPosition()
        dx = h_pos[0] - r_pos[0]
        dy = h_pos[1] - r_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        heading = 0.0 
        alpha = np.arctan2(dy, dx) - heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.array([
            np.clip(dist / 10.0, 0.0, 1.0), 
            alpha / np.pi, 
            0.0, 
            0.0, 
            self.human_emotion
        ], dtype=np.float32)
        
        return obs, {}
