import random
from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from lab3.models import Network, tensor


class Reinforce:
    def __init__(self, shape, actions, device, lr=1e-3, gamma=0.99, baseline="none"):
        self.device, self.gamma, self.baseline = device, gamma, baseline
        self.actor = Network(shape, actions).to(device)
        self.value = Network(shape, 1).to(device) if baseline == "value" else None
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr) if self.value is not None else None

    @torch.no_grad()
    def act(self, state, deterministic=False):
        logits = self.actor(tensor(state, self.device))
        return int(logits.argmax(-1)) if deterministic else int(torch.distributions.Categorical(logits=logits).sample())

    def update(self, states, actions, rewards, final_state, terminated):
        # Per la baseline di valore, un TimeLimit può essere bootstrap-pato.
        # REINFORCE puro usa il ritorno della traiettoria episodica osservata.
        tail = 0.0
        if self.value is not None and not terminated:
            with torch.no_grad():
                tail = self.value(tensor(final_state, self.device)).item()
        returns = []
        for reward in reversed(rewards):
            tail = reward + self.gamma * tail
            returns.append(tail)
        returns = torch.tensor(returns[::-1], device=self.device)
        states = torch.as_tensor(np.stack(states), device=self.device)
        actions = torch.tensor(actions, device=self.device)
        if self.value is not None:
            values = self.value(states).squeeze(-1)
            advantages = returns - values.detach()
        elif self.baseline == "standardized":
            advantages = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-8)
        else:
            advantages = returns
        distribution = torch.distributions.Categorical(logits=self.actor(states))
        policy_loss = -(distribution.log_prob(actions) * advantages).mean()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()
        value_loss = torch.zeros((), device=self.device)
        if self.value is not None:
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.value_optimizer.step()
        return dict(policy_loss=policy_loss.item(), value_loss=value_loss.item())

    def state_dict(self):
        result = {"actor": self.actor.state_dict()}
        if self.value is not None:
            result["value"] = self.value.state_dict()
        return result

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        if self.value is not None:
            self.value.load_state_dict(state["value"])


class DQN:
    def __init__(self, shape, actions, device, lr=1e-3, gamma=0.99, capacity=50000, batch_size=64,
                 tau=0.005, warmup=1000, update_every=4):
        self.device, self.gamma, self.actions = device, gamma, actions
        self.batch_size, self.tau, self.warmup, self.update_every = batch_size, tau, warmup, update_every
        self.online, self.target = Network(shape, actions).to(device), Network(shape, actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)
        self.steps = 0

    @torch.no_grad()
    def act(self, state, deterministic=False, epsilon=0.0):
        if not deterministic and random.random() < epsilon:
            return random.randrange(self.actions)
        return int(self.online(tensor(state, self.device)).argmax(-1))

    def observe(self, state, action, reward, next_state, terminated):
        # Conserviamo uint8 per CarRacing; non salviamo tensori GPU nel replay.
        self.memory.append((state.copy(), action, reward, next_state.copy(), terminated))
        self.steps += 1
        if len(self.memory) < max(self.warmup, self.batch_size) or self.steps % self.update_every:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        states = torch.as_tensor(np.stack(states), device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32, device=self.device)
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * (1 - terminated) * self.target(next_states).max(1).values
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.online.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return loss.item()

    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict()}

    def load_state_dict(self, state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])


class PPO:
    def __init__(self, shape, actions, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 update_epochs=4, minibatch=64, entropy=0.01):
        self.device, self.gamma, self.gae_lambda = device, gamma, gae_lambda
        self.clip, self.update_epochs, self.minibatch, self.entropy = clip, update_epochs, minibatch, entropy
        self.actor, self.value = Network(shape, actions).to(device), Network(shape, 1).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.value.parameters()), lr=lr)

    @torch.no_grad()
    def sample_batch(self, states):
        states = torch.as_tensor(states, device=self.device)
        distribution = torch.distributions.Categorical(logits=self.actor(states))
        actions = distribution.sample()
        return (actions.cpu().numpy(), distribution.log_prob(actions).cpu().numpy(),
                self.value(states).squeeze(-1).cpu().numpy())

    @torch.no_grad()
    def sample(self, state):
        state = tensor(state, self.device)
        distribution = torch.distributions.Categorical(logits=self.actor(state))
        action = distribution.sample()
        return int(action), distribution.log_prob(action).item(), self.value(state).item()

    @torch.no_grad()
    def act(self, state, deterministic=False):
        logits = self.actor(tensor(state, self.device))
        return int(logits.argmax(-1)) if deterministic else int(torch.distributions.Categorical(logits=logits).sample())

    def update(self, transitions):
        states, actions, log_probs, values, rewards, next_values, terminated, done = zip(*transitions)
        advantages, accumulator = [], 0.0
        for index in reversed(range(len(transitions))):
            delta = rewards[index] + self.gamma * (not terminated[index]) * next_values[index] - values[index]
            # Non propagare GAE oltre un reset; bootstrap sul TimeLimit, non su terminali.
            accumulator = delta + self.gamma * self.gae_lambda * (not done[index]) * accumulator
            advantages.append(accumulator)
        advantages = torch.tensor(advantages[::-1], dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        states = torch.as_tensor(np.stack(states), device=self.device)
        actions = torch.tensor(actions, device=self.device)
        old_log_probs = torch.tensor(log_probs, device=self.device)
        losses = []
        for _ in range(self.update_epochs):
            order = torch.randperm(len(transitions), device=self.device)
            for start in range(0, len(order), self.minibatch):
                indices = order[start:start + self.minibatch]
                distribution = torch.distributions.Categorical(logits=self.actor(states[indices]))
                ratio = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
                surrogate = torch.minimum(ratio * advantages[indices], ratio.clamp(1 - self.clip, 1 + self.clip) * advantages[indices])
                value_loss = F.mse_loss(self.value(states[indices]).squeeze(-1), returns[indices])
                policy_loss = -surrogate.mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy * distribution.entropy().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.value.parameters()), 0.5)
                self.optimizer.step()
                losses.append((policy_loss.item(), value_loss.item()))
        return dict(policy_loss=float(np.mean([item[0] for item in losses])), value_loss=float(np.mean([item[1] for item in losses])))

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "value": self.value.state_dict()}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
