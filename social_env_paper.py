"""
SocialNavEnv - Versión física honesta (paper).

Cambios respecto al entorno original:
  [B1] Condición de éxito con umbral de frenado RELATIVO a max_v (no absoluto 0.05).
  [B2] Action space normalizado a [-1, 1] + reescalado interno a la física real.
       Observación normalizada por escala; velocidades reales expuestas a la red.
  [B3] Reward de progreso SIN densificar (factor 25.0 intacto). Castigo lateral intacto.
  [B4] Speeding ticket de intrusión parametrizable: suave (eta*|v|, paper) o fijo (-20).
  [B5] Timeout -> truncated (no terminated). Emoción codificada -1/0/1.
  Curriculum de 3 fases (set_phase) para usar con un callback en train.py.

Física real del e-puck: max_v = 0.12 m/s. NO se entrena a 0.6 con escalado externo.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SocialNavEnv(gym.Env):
    def __init__(self, smooth_penalty=True, phase=3, eta=0.5, progress_weight=25.0, no_terminal_bonus=False):
        super(SocialNavEnv, self).__init__()

        self.dt = 0.1

        # --- Física real del e-puck ---
        self.max_v = 0.12       # avance máximo real (m/s)
        self.min_v = -0.06      # reversa permitida (rectificación activa)
        self.max_w = 1.0        # velocidad angular (rad/s)

        # [B4] eta para el speeding ticket suave del paper (R_seguridad = -eta*|v|)
        self.eta = eta
        self.progress_weight = progress_weight
        self.no_terminal_bonus = no_terminal_bonus
        self.smooth_penalty = smooth_penalty

        # Curriculum: 1 (estático, sin castigo), 2 (emociones aleatorias), 3 (estricto)
        self.phase = phase

        # [B1] Umbral de frenado RELATIVO. 0.15 * max_v ~= 0.018 m/s con max_v=0.12.
        # Expresado como fracción => invariante a la escala de velocidad.
        self.stop_frac = 0.15
        self.stop_threshold = self.stop_frac * self.max_v

        # [B2] Action space NORMALIZADO. El reescalado a la física ocurre en step().
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # [B2] Observación normalizada: [rho/10, alpha/pi, v/max_v, w/max_w, emotion]
        # Todas las componentes quedan aproximadamente en [-1, 1].
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.human_pose = np.array([2.0, 0.0], dtype=np.float32)

        # Emoción codificada como el paper: -1 (Ira), 0 (Neutro), 1 (Feliz)
        self.human_emotion = 0.0
        self.base_emotion = 0.0
        self.emotion_color = 'y'

        self.steps_counter = 0
        self.violation_time = 0
        self.prev_dist = 0.0

        # Velocidades reales (reescaladas) para exponer a la red
        self.v_actual = 0.0
        self.w_actual = 0.0

        self.trajectory_x = []
        self.trajectory_y = []

    # ------------------------------------------------------------------ #
    # Curriculum
    # ------------------------------------------------------------------ #
    def set_phase(self, phase):
        self.phase = phase

    # ------------------------------------------------------------------ #
    # Helpers de escala
    # ------------------------------------------------------------------ #
    def _scale_action(self, action):
        """[B2] Mapea accion normalizada [-1,1] a la fisica real."""
        raw_v = float(np.clip(action[0], -1.0, 1.0))
        raw_w = float(np.clip(action[1], -1.0, 1.0))
        # raw_v en [-1,1] -> [min_v, max_v]
        v = self.min_v + (raw_v + 1.0) * 0.5 * (self.max_v - self.min_v)
        w = raw_w * self.max_w
        v = float(np.clip(v, self.min_v, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        return v, w

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        # Spawn alineado (igual que el original)
        self.robot_pose = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        if self.phase == 1:
            # Fase 1: humano neutro fijo
            self.base_emotion = 0.0
        else:
            # Fases 2 y 3: emociones aleatorias
            self.base_emotion = random.choice([-1.0, 0.0, 1.0])
        self.human_emotion = self.base_emotion
        self._update_color()

        self.steps_counter = 0
        self.violation_time = 0
        self.v_actual = 0.0
        self.w_actual = 0.0
        self.trajectory_x = [self.robot_pose[0]]
        self.trajectory_y = [self.robot_pose[1]]

        self.prev_dist = self._get_distance()

        return self._get_obs(), {}

    def step(self, action):
        self.steps_counter += 1

        # [B2] Reescalar accion normalizada a fisica real
        v, w = self._scale_action(action)
        self.v_actual = v
        self.w_actual = w

        theta = self.robot_pose[2]
        self.robot_pose[0] += v * np.cos(theta) * self.dt
        self.robot_pose[1] += v * np.sin(theta) * self.dt
        self.robot_pose[2] += w * self.dt

        self.trajectory_x.append(self.robot_pose[0])
        self.trajectory_y.append(self.robot_pose[1])

        # Zonas proxemicas dinamicas (emocion -1/0/1)
        if self.human_emotion == -1.0:    # Ira
            inner, outer = 0.60, 0.80
        elif self.human_emotion == 0.0:   # Neutro
            inner, outer = 0.35, 0.60
        else:                             # Feliz (1.0)
            inner, outer = 0.20, 0.35

        # Curriculum Fase 1: meta amplia, sin zona estricta
        if self.phase == 1:
            inner, outer = 0.0, 1.0

        curr_dist = self._get_distance()

        dx = self.human_pose[0] - self.robot_pose[0]
        dy = self.human_pose[1] - self.robot_pose[1]
        alpha = (np.arctan2(dy, dx) - self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

        if curr_dist < inner:
            self.violation_time += 1
        else:
            self.violation_time = max(0, self.violation_time - 1)

        reward = 0.0
        terminated = False
        truncated = False

        # Costes densos basales (intactos respecto al original)
        reward -= 0.01
        reward -= abs(self.robot_pose[1]) * 2.0        # [B3] castigo lateral: NO se quita
        reward += (1.0 - abs(alpha)) * 1.0

        # ---------------- LOGICA DE ZONAS ----------------
        if curr_dist > outer:
            # [B3] Progreso SIN densificar
            progress = self.prev_dist - curr_dist
            if progress > 0:
                reward += progress * self.progress_weight
            else:
                reward -= 0.5

        elif inner <= curr_dist <= outer:
            # Zona objetivo (parking)
            reward += 2.0
            mid = (inner + outer) / 2.0
            dist_from_center = abs(curr_dist - mid)
            band_half_width = (outer - inner) / 2.0
            centering_score = 1.0 - (dist_from_center / band_half_width)

            # [B1] Umbral de frenado RELATIVO a max_v
            if abs(v) < self.stop_threshold:
                if not self.no_terminal_bonus:
                    final_score = 100.0 + (50.0 * centering_score)
                    if self.human_emotion == 1.0:
                        final_score *= 1.5
                    reward += final_score
                terminated = True
            else:
                reward -= 0.5 * abs(v)

        elif curr_dist < inner:
            # [B4] Penalizacion de intrusion: suave (paper) o fija
            if self.phase >= 3:
                if self.smooth_penalty:
                    # R_seguridad = -eta * |v|  (gradiente continuo, version paper)
                    reward -= self.eta * abs(v)
                else:
                    # Castigo fijo (acantilado) -- solo para ablacion
                    reward -= 20.0
                    if v > 0:
                        reward -= 5.0
                    if v < -0.05:
                        reward += 5.0
            elif self.phase == 2:
                reward -= 2.0
            # Fase 1: sin castigo por zona

        # Colision fisica (solo penaliza fuerte en fase estricta)
        if curr_dist < 0.10 and self.phase >= 3:
            reward -= 100.0
            terminated = True

        # Fuera de limites
        if abs(self.robot_pose[1]) > 5.0 or abs(self.robot_pose[0]) > 5.0:
            reward -= 50.0
            terminated = True

        if curr_dist > 7.0:
            reward -= 50.0
            terminated = True

        # [B5] Timeout -> truncated, NO terminated
        if self.steps_counter > 300:
            truncated = True

        self.prev_dist = curr_dist

        # info para evaluacion / logging
        is_success = bool(
            terminated
            and (inner <= curr_dist <= outer)
            and (abs(v) < self.stop_threshold)
        )
        info = {
            "is_success": is_success,
            "dist": curr_dist,
            "inner": inner,
            "outer": outer,
            "v": v,
            "emotion": self.human_emotion,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _update_color(self):
        if self.human_emotion == -1.0:
            self.emotion_color = 'r'
        elif self.human_emotion == 0.0:
            self.emotion_color = 'y'
        else:
            self.emotion_color = 'g'

    def _get_distance(self):
        return np.sqrt((self.human_pose[0] - self.robot_pose[0]) ** 2 +
                       (self.human_pose[1] - self.robot_pose[1]) ** 2)

    def _get_obs(self):
        """[B2] Observacion normalizada por escala. MISMA funcion en train y eval."""
        dx = self.human_pose[0] - self.robot_pose[0]
        dy = self.human_pose[1] - self.robot_pose[1]
        rho = np.sqrt(dx ** 2 + dy ** 2)
        alpha = np.arctan2(dy, dx) - self.robot_pose[2]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        return np.array([
            rho / 10.0,
            alpha / np.pi,
            self.v_actual / self.max_v,
            self.w_actual / self.max_w,
            self.human_emotion,
        ], dtype=np.float32)

    def render(self):
        pass