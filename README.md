# Bridging the Gap Between Soft RL and GFlowNets


Official code for the project [Bridging the Gap Between Soft RL and GFlowNets](https://drive.google.com/file/d/1Nc90rokaaxV5gHCdVAmPkv3IG5RcDpTH/view?usp=sharing). 

Ayhan Suleymanzade, Zahra Bayramli

This repository is build upon the related work [GFlowNets as Entropy-Regularized RL](https://github.com/d-tiapkin/gflownet-rl)

## Installation

- Create conda environment:

```sh
conda create -n gflownet-rl python=3.10
conda activate gflownet-rl
```

- Install PyTorch with CUDA. For our experiments we used the following versions:

```sh
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can change `pytorch-cuda=11.8` with `pytorch-cuda=XX.X` to match your version of `CUDA`.

- Install core dependencies:

```sh
pip install -r requirements.txt
```

## Hypergrids

Code for this part heavily utlizes library `torchgfn` (https://github.com/GFNOrg/torchgfn).

Path to configurations (utlizes `ml-collections` library):

- General configuration: `hypergrid/experiments/config/general.py`
- Algorithm: `hypergrid/experiments/config/algo.py`
- Environment: `hypergrid/experiments/config/hypergrid.py`

List of available algorithms:
- GFlowNets Baselines: `db`, `tb`, `subtb` from `torchgfn` library;
- Soft RL Baselines: `soft_dqn`, `munchausen_dqn`, `sac`.
- Our Algorithms: `learnable_dqn`, `lambda_dqn`, `pcl_dqn`, `pcl_ms_dqn`

Example of running the experiment on environment with `height=20`, `ndim=4` with `standard` rewards, seed `3` on the algorithm `lambda_dqn`.
```bash
    python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:lambda_dqn --env.height 20 --env.ndim 4
```

## Bit sequences

Examples of running `TB`, `DB` and `SubTB` baselines for word length `k=8`:

```
python bitseq/run.py --objective tb --k 8 --learning_rate 0.002
```

```
python bitseq/run.py --objective db --k 8 --learning_rate 0.002
```

```
python bitseq/run.py --objective subtb --k 8 --learning_rate 0.002 --subtb_lambda 1.9
```

Example of running `SoftDQN`:

```
python bitseq/run.py --objective softdqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

Example of running `MunchausenDQN`:

```
python bitseq/run.py --objective softdqn --m_alpha 0.15 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

Example of running `LearnableDQN`:

```
python bitseq/run.py --objective learnable_dqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

Example of running `LambdaDQN`:

```
python bitseq/run.py --objective lambda_dqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 --lambda_dist "Gamma"
```

Example of running `PCLDQN`:

```
python bitseq/run.py --objective pcl_dqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

Example of running `PCL_MS_DQN`:

```
python bitseq/run.py --objective pcl_ms_dqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 --v_learning_rate 0.001
```

## Results

You can find all the results in the following directories:

- `/hypergrid/grid_results`
- `/bitseq/bitseq_results`

These directories correspond to the Hypergrid and Bit sequences experiments, respectively.

### Hypergrid Results

The columns in the Hypergrid results files are as follows:
1. **First Column:** KL Divergence
2. **Second Column:** L1 Distance
3. **Third Column:** Timestep

### Bit Sequences Results

The columns in the Bit Sequences results files are as follows:
1. **First Column:** Number of Modes Captured
2. **Second Column:** Spearman Correlation
3. **Third Column:** Timestep




