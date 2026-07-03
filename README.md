# Emotion-Aware Social Navigation via Deep Reinforcement Learning

This repository contains the official codebase and simulation environment for the paper on **Emotion-Aware Social Navigation using Deep Reinforcement Learning**. 

This repository provides everything needed to train, evaluate, and deploy a Proximal Policy Optimization (PPO) agent capable of approaching humans while respecting social distances derived from human emotional states (Angry, Neutral, Happy).

---

## 📖 Overview

The core contribution of this work is the development of a hybrid continuous reward formulation featuring a **Stop-to-Win** mechanism, coupled with a **Curriculum Learning** strategy. This enables the agent to safely approach humans and physically stop at dynamically determined social comfort boundaries, preventing typical failure modes like "reward hacking" (e.g., indefinite orbiting).

### Key Features
- **Emotion-Conditioned Comfort Zones**: Social distances dynamically adapt based on the simulated human's emotion (e.g., 0.20m for Happy, 0.60m for Angry).
- **Stop-to-Win Architecture**: The agent is explicitly trained to decelerate and come to a complete halt ($v < 0.018$ m/s) inside the target zone, reversing the incentive to indefinitely exploit dense rewards.
- **Perfect Spatial Generalization**: By utilizing relative polar coordinates ($\rho, \alpha$), the model guarantees $360^\circ$ spatial robustness regardless of the initial spawn angle.

---

## 🛠️ Installation & Setup

To run this code locally, you will need Python 3.8+ and the following dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/social-navigation-rl.git
cd social-navigation-rl/paper

# Install dependencies
pip install -r requirements.txt
```
*(Dependencies mainly include `gymnasium`, `stable-baselines3`, `numpy`, `matplotlib`, and `pandas`).*

---

## 🚀 How to Train the Agent

The training process uses a **3-Phase Curriculum Learning** approach to ensure stable convergence. The agent learns progressively:
1. **Phase 1 (Static)**: No spatial constraints; the agent learns to move towards the human.
2. **Phase 2 (Random Emotions)**: The comfort zones become dynamic based on emotions.
3. **Phase 3 (Strict Stop)**: Speeding penalties are introduced, and the agent must strictly respect the stopping thresholds to succeed.

To train the model from scratch, simply run:

```bash
python train_paper.py
```

- **Output**: The script will periodically save the best performing model inside the `models/` directory (e.g., `ppo_social_nav_best.zip`).
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3.
- **Timesteps**: Default is 1,000,000 steps.

---

## 📊 Evaluation & Metrics

Once you have a trained model, you can rigorously evaluate its performance. Our evaluation suite tests the model under a **$360^\circ$ Generalization Protocol** (spawning the robot at random angles around the human) and measures four critical metrics:
- **Success Rate (%)**: Reaching the comfort zone and stopping completely.
- **SII (%)**: Spatial Intrusion Index (percentage of steps violating the human's personal space).
- **Collisions (%)**: Physical crashes ($< 0.10$ m).
- **Jerk ($m/s^3$)**: Smoothness of the trajectory.

To run the evaluation:

```bash
python check_results_paper.py
```
This will output a Markdown table with the metrics split by emotion, exactly as reported in the paper.

---

## 📈 Visualizing Trajectories

If you want to visualize the learned control law and the physical trajectory of the robot as it approaches the human, you can use the plotting scripts:

```bash
# To plot trajectories from random 360-degree spawns
python plot_random_spawns_paper.py

# To generate high-resolution plots for specific episodes (used in the paper)
python generate_trajectory_plots_paper.py
```
This will generate PDF/PNG vector plots showing the robot's path, the comfort zones, and its velocity profile.

---

## 🤖 Webots Physical Simulation (e-puck)

This project bridges the gap between 2D kinematic simulation and physical reality. We include integration scripts for the **Webots** robotics simulator, allowing the trained policy to control a realistic e-puck robot physics model.

To run the trained policy inside Webots:
1. Open the `.wbt` world file in Webots.
2. Ensure your Webots Python API is correctly linked.
3. Run the interactive deployment script:
```bash
python run_interactive_webots_paper.py
```

---

## 📂 Repository Structure (Paper Code)

| File | Description |
|------|-------------|
| `social_env_paper.py` | The main Gymnasium environment featuring the Stop-to-Win logic and Curriculum rules. |
| `train_paper.py` | The PPO training script orchestrating the Curriculum Learning callback. |
| `check_results_paper.py` | Comprehensive evaluation script calculating Success, SII, and Jerk. |
| `stress_test_paper.py` | Injects Gaussian sensorimotor noise to evaluate model resilience. |
| `webots_env_paper.py` | Wrapper for the Webots simulator interface. |

---
*For any questions regarding the methodology or reproducibility, please refer to the corresponding sections in the main thesis document or open an issue.*
