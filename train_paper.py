"""
Entrenamiento - Versión física honesta (paper).

Cambios respecto al train original:
  [B3] Curriculum fijo ASIMETRICO: 200k / 300k / 500k = 1M total (Fase 3 = mitad).
       Transiciones alineadas a multiplos de n_steps (2048) para curvas limpias.
  [B4] log_std_init bajo (-1.0): politica menos ruidosa, clave para control fino de frenado.
  [B5] Seed totalmente parametrizable (modelo + env + action_space).
  LR con decaimiento lineal (3e-4 -> 0), igual que la Tabla 1 del paper.

USO:
  Fase de caza  -> SEED = 42 (un solo seed hasta lograr comportamiento consistente).
  Fase evidencia -> repetir con SEEDS = [123, 456, 789, 1011] tras CONGELAR la config.
"""

import os
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from social_env_paper import SocialNavEnv

import sys
# ===================== CONFIGURACION =====================
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42      # <-- fase de caza: dejar 42 fijo
SMOOTH_PENALTY = True          # True = speeding ticket suave (paper). False = ablacion fija.

# Presupuesto del curriculum (alineado a n_steps=2048)
PHASE1_STEPS = 198656          # ~200k  (97 * 2048)
PHASE2_STEPS = 299008          # ~300k  (146 * 2048)  -> acumulado ~500k
PHASE3_STEPS = 499712          # ~500k  (244 * 2048)  -> acumulado ~1M
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS

PHASE2_START = PHASE1_STEPS
PHASE3_START = PHASE1_STEPS + PHASE2_STEPS
# ========================================================


class CurriculumCallback(BaseCallback):
    """Avance de fase por pasos fijos asimetricos."""

    def __init__(self, p2_start, p3_start, verbose=0):
        super().__init__(verbose)
        self.p2_start = p2_start
        self.p3_start = p3_start
        self.phase = 1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.p2_start and self.phase != 1:
            self.training_env.env_method('set_phase', 1)
            self.phase = 1
        elif self.p2_start <= t < self.p3_start and self.phase != 2:
            self.training_env.env_method('set_phase', 2)
            self.phase = 2
            if self.verbose:
                print(f"[{t}] --> Fase 2: emociones aleatorias.")
        elif t >= self.p3_start and self.phase != 3:
            self.training_env.env_method('set_phase', 3)
            self.phase = 3
            if self.verbose:
                print(f"[{t}] --> Fase 3: dinamico + castigos estrictos.")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """LR lineal: progress_remaining va de 1.0 (inicio) a 0.0 (final)."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


if __name__ == "__main__":
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Entorno (arranca en Fase 1). Seed del action_space para reproducibilidad.
    env = SocialNavEnv(smooth_penalty=SMOOTH_PENALTY, phase=1)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    # [B4] log_std_init bajo => exploracion menos agresiva (mejor para parking/frenado fino)
    policy_kwargs = dict(log_std_init=-1.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu',
    )

    callback = CurriculumCallback(PHASE2_START, PHASE3_START, verbose=1)

    print("=" * 66)
    print(f" ENTRENAMIENTO FISICA HONESTA | seed={SEED} | smooth={SMOOTH_PENALTY}")
    print(f" Total: {TOTAL_STEPS} pasos  (F1={PHASE1_STEPS} / F2={PHASE2_STEPS} / F3={PHASE3_STEPS})")
    print("=" * 66)

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    out_path = f"{models_dir}/ppo_social_nav_seed{SEED}"
    model.save(out_path)
    print(f"Modelo guardado en {out_path}.zip")