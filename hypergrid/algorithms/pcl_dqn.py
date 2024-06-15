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


class PCLDQNGFlowNet(GFlowNet):
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

    def get_scores(
        self, transitions: Transitions
    ):
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        q_s = self.q(states) / self.entropy_coeff
        q_s[~states.forward_masks] = -float("inf")

        # change actions.tensor device to q_s device
        actions.tensor = actions.tensor.to(q_s.device)
        # preds = torch.gather(q_s, 1, actions.tensor).squeeze(-1)

        # preds to be in V(s) shape zeros
        # preds = torch.zeros(q_s.shape[0], device=q_s.device)
        preds = torch.gather(q_s, 1, actions.tensor).squeeze(-1)
        targets = torch.zeros_like(preds)

        # valid_next_states = transitions.next_states
        
        # import pdb; pdb.set_trace()
        try:
            valid_next_states = transitions.next_states[~transitions.is_done]
        except RuntimeError as e:
            print("Error occurred!")
            print("Next state device: ", transitions.next_states.device)
            print("Is done device: ", transitions.is_done.device)
            raise e  # Optionally re-raise the error to stop the program or handle it as needed
        
        non_exit_actions = actions[~actions.is_exit]

        module_output = self.pb(valid_next_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        #import pdb; pdb.set_trace()

        policy_s = torch.exp(q_s)
        policy_sum = policy_s.sum(dim=-1, keepdim=True)


        policy_s_a = torch.gather(policy_s/ policy_sum, 1, actions.tensor).squeeze(-1)
        
        log_policy_s_a = torch.log(policy_s_a + 1e-9)

        import pdb; pdb.set_trace()

       
        # log_policy_s_a = torch.gather(log_policy_s, 1, actions.tensor).squeeze(-1)

        # log_policy_s_a = log_policy_s[:, 0]
        valid_v = self.entropy_coeff * torch.logsumexp(
                    q_s / self.entropy_coeff, dim=-1
                ).squeeze(-1)
        

        with torch.no_grad():
            q_s_target = self.q_target(states)
            q_s_target[~states.forward_masks] = -float("inf")
            q_sn_target = self.q_target(valid_next_states)
            q_sn_target[~valid_next_states.forward_masks] = -float("inf")
            valid_v_target_next = self.entropy_coeff * torch.logsumexp(
                    q_sn_target / self.entropy_coeff, dim=-1
                ).squeeze(-1) 
            
          
        ## sample action with log_pb and use it to calculate target
    

        targets[~valid_transitions_is_done] = valid_log_pb_actions + valid_v_target_next - valid_v[~valid_transitions_is_done] - self.entropy_coeff * log_policy_s_a[~valid_transitions_is_done] 


        # preds = self.entropy_coeff * log_policy_s_a[~valid_transitions_is_done] 

        assert transitions.log_rewards is not None
    
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        # import pdb; pdb.set_trace()
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ] - valid_v[valid_transitions_is_done] - self.entropy_coeff * log_policy_s_a[valid_transitions_is_done]


        if self.munchausen_alpha > 0:
            with torch.no_grad():
                q_s_target = self.q_target(states) / self.entropy_coeff
                q_s_target[~states.forward_masks] = -float("inf")
                v_s_target = torch.logsumexp(q_s_target, dim=-1).squeeze(-1)
                q_sa_target = torch.gather(q_s_target, 1, actions.tensor).squeeze(-1)
                log_pf_target = (q_sa_target - v_s_target)
                penalty = (self.entropy_coeff * log_pf_target).clamp(
                    min=self.munchausen_l0,
                    max=0.
                )
            targets += self.munchausen_alpha * penalty

        scores = targets
        return scores

    def loss(self, transitions: Transitions) -> TT[0, float]:
        _, _, scores = self.get_scores(transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
