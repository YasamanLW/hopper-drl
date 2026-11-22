# Hopper-v5 Reinforcement Learning with PPO

A Deep Reinforcement Learning project that trains an autonomous agent to control the Hopper robot in the MuJoCo physics simulation environment using Proximal Policy Optimization (PPO).

## Project Overview

The goal is to train an agent to make the Hopper robot move forward while maintaining balance and maximizing its cumulative reward.

The project uses:
- **Gymnasium** for the reinforcement learning environment.
- **MuJoCo** for physics simulation.
- **Stable-Baselines3** for the PPO implementation.
- **PyTorch** as the deep learning framework.

## Algorithm

**Proximal Policy Optimization (PPO)** is a policy-gradient reinforcement learning algorithm that improves an agent's policy through iterative updates while limiting excessively large policy changes.

The agent uses an MLP (Multi-Layer Perceptron) policy to process observations and select continuous actions.

## Training Configuration

| Parameter | Value |
|---|---|
| Environment | Hopper-v5 |
| Algorithm | PPO |
| Policy | MlpPolicy |
| Total Training Timesteps | 110,000 |
| Random Seed | 42 |
| Framework | Stable-Baselines3 |

Training is performed in three stages:
- 10,000 timesteps
- 50,000 timesteps
- 50,000 timesteps

The trained model is saved as `hopper_ppo_final.zip`.

## Evaluation Results

The trained model was evaluated over five episodes.

| Metric | Result |
|---|---:|
| Number of Episodes | 5 |
| Average Reward | 746.39 |
| Minimum Reward | 741.90 |
| Maximum Reward | 750.34 |

These results represent the evaluation run using deterministic policy predictions.

## Project Structure

```text
hopper-drl/
│
├── Hopper-DRL-YasamanFallahi.ipynb
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YasamanLW/hopper-drl.git
cd hopper-drl
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python train.py
```

After training, the model will be saved as:

```text
hopper_ppo_final.zip
```

## Evaluation

Run the evaluation script:

```bash
python evaluate.py
```

The script loads the trained model, runs the Hopper environment, and reports the reward obtained in each episode.

## Author

**Yasaman Fallahi**

GitHub: [@YasamanLW](https://github.com/YasamanLW)
