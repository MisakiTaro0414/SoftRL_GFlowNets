import torch
from gfn.modules import DiscretePolicyEstimator, GFNModule
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.gflownet import GFlowNet
from gfn.states import States
from gfn.samplers import Sampler


class RescaledDPE(DiscretePolicyEstimator):
    def __init__(
            self,
            module: DiscretePolicyEstimator,
            scaling: float = 1.
    ):
        super().__init__(
            module.env,
            module.module,
            module._forward,
            module._greedy_eps,
            module.temperature,
            module.sf_bias,
            module.epsilon
        )
        self.scaling = scaling

    def forward(self, states: States):
        return super().forward(states) / self.scaling


class PCL_MS_DQNGFlowNet(GFlowNet):
    def __init__(
        self,
        q: GFNModule,
        q_target: GFNModule,
        pb: GFNModule,
        on_policy: bool = False,
        is_double: bool = False,
        entropy_coeff: float = 1.,
        munchausen_alpha: float = 0.,
        munchausen_l0: float = -2.
    ):
        super().__init__()
        self.q = q
        self.pb = pb
        self.on_policy = on_policy
        self.is_double = is_double

        self.q_target = q_target
        self.q_target.load_state_dict(self.q.state_dict())

        self.entropy_coeff = entropy_coeff
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_l0 = munchausen_l0

        self.v_sum = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def sample_trajectories(self, n_samples: int = 1000) -> Trajectories:
        sampler = Sampler(estimator=RescaledDPE(
                self.q, scaling=self.entropy_coeff
            )
        )
        trajectories = sampler.sample_trajectories(n_trajectories=n_samples)
        return trajectories

    def update_q_target(self, tau=0.05):
        if tau == 1.:
            self.q_target.load_state_dict(self.q.state_dict())
            return
        with torch.no_grad():
            for param, target_param in zip(
                self.q.parameters(), self.q_target.parameters()
            ):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau,
                          out=target_param.data)

    def get_scores(self, trajectories: Trajectories):
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if trajectories.is_backward:
            raise ValueError("Backward transitions are not supported")
        # states = trajectories.states
        # actions = trajectories.actions

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

        
        if valid_states.batch_shape != tuple(valid_actions.batch_shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")


        ## POLICY CALCULATION
        module_output = self.q(valid_states)
        valid_log_policy_s_a = self.q.to_probability_distribution(
            valid_states, module_output
        ).log_prob(valid_actions.tensor)

        log_policy_s_a_trajectories = torch.full_like(
            trajectories.actions.tensor[..., 0],
            fill_value=0.0,
            dtype=torch.float,
        )
        log_policy_s_a_trajectories[~trajectories.actions.is_dummy] = valid_log_policy_s_a

        total_log_policy_s_a = log_policy_s_a_trajectories.sum(dim=0)



        ### REWARD CALCULATION
        non_initial_valid_states = valid_states[~valid_states.is_initial_state]
        non_exit_valid_actions = valid_actions[~valid_actions.is_exit]

        module_output = self.pb(non_initial_valid_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            non_initial_valid_states, module_output
        ).log_prob(non_exit_valid_actions.tensor)

        log_pb_trajectories = torch.full_like(
            trajectories.actions.tensor[..., 0],
            fill_value= 0.0,
            dtype=torch.float,
        )
        log_pb_trajectories_slice = torch.full_like(
            valid_actions.tensor[..., 0], fill_value= 0.0, dtype=torch.float
        )
        log_pb_trajectories_slice[~valid_actions.is_exit] = valid_log_pb_actions
        log_pb_trajectories[~trajectories.actions.is_dummy] = log_pb_trajectories_slice

        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0) # total reward

        final_rewards = trajectories.log_rewards


        targets = self.v_sum + self.entropy_coeff * total_log_policy_s_a - total_log_pb_trajectories - final_rewards

        scores = targets
        return scores

    def loss(self, transitions: Transitions) -> TT[0, float]:
        _, _, scores = self.get_scores(transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories
