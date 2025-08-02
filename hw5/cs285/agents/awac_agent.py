from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            # DONE
            qa_values = self.target_critic(next_observations)

            # Use the actor to compute a critic backup
            # next_action_logits=self.actor(next_observations) # shape:[batch_size,num_actions]
            # next_action_dist=torch.distributions.categorical.Categorical(logits=next_action_logits)
            next_action_dist = self.actor(next_observations)
            next_qs = torch.sum(next_action_dist.probs * qa_values, dim=-1)

            # TODO(student): Compute the TD target
            target_values = rewards+self.discount*next_qs*(1-dones.float())

        # TODO(student): Compute Q(s, a) and loss similar to DQN
        q_values = torch.gather(self.critic(observations), -1, torch.unsqueeze(actions.long(), 1)).squeeze()
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        # DONE
        qa_values= self.critic(observations)    
        q_values = torch.gather(qa_values, -1, torch.unsqueeze(actions.long(), 1))
        values = torch.sum(action_dist.probs * qa_values, dim=-1, keepdim=True) if action_dist else qa_values.mean(dim=-1, keepdim=True)

        advantages = q_values - values
        # clip the advantages' value
        advantages = torch.clamp(advantages, min=-3, max=3)
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        # DONE
        # actions_logits = self.actor(observations)
        # action_dist = torch.distributions.categorical.Categorical(logits=actions_logits)
        action_dist = self.actor(observations)
        loss = -torch.mean(action_dist.log_prob(actions) * torch.exp(self.compute_advantage(observations, actions, action_dist)/self.temperature))

        self.actor_optimizer.zero_grad()
        loss.backward()
        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
