# Reinforcement Learning Chess

## Parameters
| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `n_episodes` | `int` | Number of episodes to train | `100` |
| `state_dim` | `int` | Dimension of states | `64` |
| `action_space` | `int` | Dimension of action space | `4096` |
| `k_steps` | `int` | Number of unroll steps for training | `5` |
| `n_simulations` | `int` | Number of MCTS simulations per move | `800` |
| `max_game_length` | `int` | Maximum length of a game | `1000` |
| `c_puct` | `float` | PUCT exploration constant for MCTS | `1.5` |
| `batch_size` | `int` | Training batch size | `8` |
| `learning_rate` | `float` | Learning rate for optimizer | `1e-3` |
| `seed` | `int` | Random seed for reproducibility | `2025` |
| `noise` | `bool` | Add Dirichlet noise to root node priors | `True/False` |
| `noise_alpha` | `float` | Dirichlet noise alpha parameter (concentration) | `0.3` |
| `noise_epsilon` | `float` | Dirichlet noise epsilon parameter (mixing ratio) | `0.25` |
| `save_every` | `int` | Save model checkpoint every n steps | `10000` |
| `verbose` | `bool` | Enable verbose output during training | `True/False` |

## Train Model
```bash
python train.py --noise --n_episodes 1000000
```

## Test Model
```bash
python test.py
```