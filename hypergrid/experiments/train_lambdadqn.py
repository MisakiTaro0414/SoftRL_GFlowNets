import numpy as np
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from tqdm import tqdm, trange


from algorithms import LambdaDQNGFlowNet, TorchRLReplayBuffer
from torchtyping import TensorType as TT
from algorithms.model import DiscretePolicyEstimator
from experiments.utils import validate
from gfn.utils.modules import NeuralNet, DiscreteUniform
from gfn.env import Env
from collections import Counter
from typing import Dict, Optional

from gfn.gflownet import GFlowNet
from gfn.states import States
from ml_collections.config_dict import ConfigDict


def train_lambdadqn(
        env: Env,
        experiment_name: str,
        general_config: ConfigDict,
        algo_config: ConfigDict):

    if algo_config.uniform_pb:
        experiment_name += '_uniform-pb'
    else:
        experiment_name += '_learnt-pb'

    if algo_config.update_frequency > 1:
        experiment_name += f"_freq={algo_config.update_frequency}"

    if algo_config.is_double:
        experiment_name += '_Double'

    if algo_config.replay_buffer.replay_buffer_size > 0:
        if algo_config.replay_buffer.prioritized:
            experiment_name += '_PER'
        else:
            experiment_name += '_ER'

    if algo_config.loss_type != 'MSE':
        experiment_name += f'_loss_type={algo_config.loss_type}'

    if algo_config.munchausen.alpha > 0:
        experiment_name += f"_M_alpha={algo_config.munchausen.alpha}"

    use_wandb = len(general_config.wandb_project) > 0
    # use_wandb = False
    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim + 1,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    pf_module.to(env.device)

    if algo_config.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)
    else:
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim + 1,
            output_dim=env.n_actions - 1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )
    pb_module.to(env.device)

    pf_estimator = DiscretePolicyEstimator(
        env=env, module=pf_module, forward=True)
    pf_estimator.to(env.device)
    pb_estimator = DiscretePolicyEstimator(
        env=env, module=pb_module, forward=False)
    pb_estimator.to(env.device)

    pf_target = NeuralNet(
        input_dim=env.preprocessor.output_dim + 1,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    pf_target.to(env.device)

    pf_target_estimator = DiscretePolicyEstimator(
        env=env, module=pf_target, forward=True)
    pf_target_estimator.to(env.device)

    replay_buffer_size = algo_config.replay_buffer.replay_buffer_size

    # sample from 0-1 for entropy coefficient from normal distribution
    gflownet = LambdaDQNGFlowNet(
        q=pf_estimator,
        q_target=pf_target_estimator,
        pb=pb_estimator,
        on_policy=True if replay_buffer_size == 0 else False,
        is_double=algo_config.is_double,
        munchausen_alpha=algo_config.munchausen.alpha,
        munchausen_l0=algo_config.munchausen.l0
    )
    gflownet.to(env.device)

    
    replay_buffer = None
    if replay_buffer_size > 0:
        replay_buffer = TorchRLReplayBuffer(
            env,
            replay_buffer_size=replay_buffer_size,
            prioritized=algo_config.replay_buffer.prioritized,
            alpha=algo_config.replay_buffer.alpha,
            beta=algo_config.replay_buffer.beta,
            batch_size=algo_config.replay_buffer.batch_size
        )

    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items()
                if ("q_target" not in k)
            ],
            "lr": algo_config.learning_rate,
        }
    ]

    if algo_config.loss_type == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif algo_config.loss_type == 'Huber':  # Used for gradient clipping
        loss_fn = torch.nn.HuberLoss(reduction='none', delta=1.0)
    else:
        raise NotImplementedError(
            f"{algo_config.loss_type} loss is not supported"
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history = [], [], []

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs
    # make entropy_coeff a learnable parameter based on Q(s,a)
    # entropy_coeff = torch.nn.Parameter(torch.tensor(1.0))
    entropy_coeff_net = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=1,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )

    entropy_coeff_net.to(env.device)

    for iteration in trange(n_iterations):
        progress = float(iteration) / n_iterations
        # entropy_coeff = torch.distributions.Uniform(0, 2).sample().item()
        # entropy_coeff = torch.distributions.Normal(0.5, 0.1).sample().item()
        entr
        trajectories = gflownet.sample_trajectories(n_samples=general_config.n_envs, entropy_coeff=entropy_coeff)
        training_samples = gflownet.to_training_samples(trajectories)



        if replay_buffer is not None:
            with torch.no_grad():
                # For priortized RB
                if replay_buffer.prioritized:
                    scores = gflownet.get_scores(training_samples, entropy_coeff)
                    td_error = loss_fn(scores, torch.zeros_like(scores))
                    replay_buffer.add(training_samples, td_error)
                    # Annealing of beta
                    replay_buffer.update_beta(progress)
                else:
                    replay_buffer.add(training_samples)

            if iteration > algo_config.learning_starts:
                training_objects, rb_batch = replay_buffer.sample()            
                scores = gflownet.get_scores(training_objects, entropy_coeff)
        else:
            training_objects = training_samples
            scores = gflownet.get_scores(training_objects, entropy_coeff)

        if iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0:
            optimizer.zero_grad()
            td_error = loss_fn(scores, torch.zeros_like(scores))
            if replay_buffer is not None and replay_buffer.prioritized:
                replay_buffer.update_priority(rb_batch, td_error.detach())
            loss = td_error.mean()
            loss.backward()
            optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"states_visited": states_visited}
        if iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0:
            if iteration % algo_config.target_network_frequency == 0:
                gflownet.update_q_target(algo_config.tau)
            to_log.update({"loss" : loss.item()})

        if use_wandb:
            wandb.log(to_log, step=iteration)

        if (iteration + 1) % general_config.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                general_config.validation_samples,
                visited_terminating_states,
            )

            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")

            kl_history.append(to_log["kl_dist"])
            l1_history.append(to_log["l1_dist"])
            nstates_history.append(to_log["states_visited"])

        if (iteration + 1) % 1000 == 0:
            np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
            np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
            np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))

    np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
    np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
    np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))



def get_terminating_state_dist_pmf(env: Env, states: States) -> TT["n_states", float]:
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(states_indices)
    counter_list = [
        counter[state_idx] if state_idx in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: Env,
    gflownet: GFlowNet,
    n_validation_samples: int = 20000,
    visited_terminating_states: Optional[States] = None,
    entropy_coeff: float = 1.0,
) -> Dict[str, float]:
    """Evaluates the current gflownet on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the gflownet on.
        gflownet: The gflownet to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns: A dictionary containing the l1 validation metric. If the gflownet
        is a TBGFlowNet, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the gflownet
        return {}

    logZ = None
    if visited_terminating_states is None:
        terminating_states = gflownet.sample_terminating_states(n_validation_samples, entropy_coeff)
    else:
        terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
    
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    kl_dist = (true_dist_pmf * torch.log(true_dist_pmf / (final_states_dist_pmf + 1e-9))).sum().item()
    validation_info = {"l1_dist": l1_dist, "kl_dist": kl_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info