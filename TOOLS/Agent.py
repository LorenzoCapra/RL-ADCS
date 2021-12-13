import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output
import gym
from TOOLS.Networks import ActorNetwork, Critic, init_weights
from Utils import get_gae, trajectories_data_generator, render_simulation


# DEFINE THE PPO AGENT:
# First we create the memory:
class Memory:
    """Storing the memory of the trajectory (s, a, r ...)."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []


class PPOAgent(object):
    """ IMPLEMENTATION OF THE PPO AGENT FOR CONTINUOUS ACTION SPACES.

        Parameters:
        env (gym.Env): gym environment for training.
        gamma (float): coef for discount factor.
        lamda (float): coef for general adversarial estimator (GAE).
        entropy_coef (float): coef of weighting entropy in objective loss.
        epsilon (float): clipping range for actor objective loss.
        actor_lr (float): learning rate for actor optimizer.
        critic_lr (float): learning rate for critic optimizer.
        value_range (float): clipping range for critic objective loss.
        rollout_len (int): num t-steps per one rollout.
        total_rollouts (int): num rollouts.
        update_timesteps (int): num of timesteps after which updating the networks
        num_epochs (int): num weights update iteration for one policy update.
        batch_size (int): data batch size for weights updating
        solved_reward (float): desired reward.
        plot_interval (int): number of episodes after which it plots the learning curves
        regulate_entropy (bool, int): defines if the entropy coefficient is static or evolving
        action_feasible (bool, float, float): check if the action sequence is feasible -> bool
                                              reward if action not feasible -> float 1
                                              max angle possible -> float 2
        save (bool): save or not the model

        AUTHOR: LORENZO CAPRA - POLITECNICO DI MILANO
    """

    def __init__(
            self,
            env: gym.Env,
            gamma: float,
            lamda: float,
            entropy_coef: float,
            epsilon: float,
            rollout_len: int,
            total_rollouts: int,
            update_timesteps: int,
            num_epochs: int,
            batch_size: int,
            is_evaluate: bool,
            value_range: float = 0.5,
            solved_reward: int = None,
            actor_lr: float = 1e-4,
            critic_lr: float = 5e-4,
            plot_interval: int = 100,
            regulate_entropy: (bool, int) = (False, 1),
            save: bool = False
    ):
        """
        Initialization.
        """
        # Initialize the environment
        self.env = env

        # Set the hyperparameters of the algorithm
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coef = entropy_coef
        self.regulate_entropy = regulate_entropy[0]
        self.entropy_start = entropy_coef  # Initial and final value of the
        self.entropy_end = entropy_coef / regulate_entropy[1]  # entropy coefficient evolution
        self.epsilon = epsilon
        self.value_range = value_range

        # Set the hyperparameters of the simulation and training
        self.rollout_len = rollout_len
        self.total_rollouts = total_rollouts
        self.update_timesteps = update_timesteps
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = ActorNetwork(self.obs_dim, self.action_dim).apply(init_weights)
        self.critic = Critic(self.obs_dim).apply(init_weights)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Memory of trajectory (s, a, r ...)
        self.memory = Memory()

        # Initialize lists for scores
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.scores = []
        self.avg_score = []
        self.trajectory = []

        # Initialize remaining parameters
        self.is_evaluate = is_evaluate
        self.solved_reward = solved_reward
        self.plot_interval = plot_interval
        self.save = save

    def _get_action(self, state: np.ndarray):  # -> float
        """
        Get action from actor, and if not test -
        get state value from critic, collect elements of trajectory.
        """
        state = torch.FloatTensor(state)
        action, dist = self.actor.forward(state)
        selected_action = dist.mean if self.is_evaluate else action
        # selected_action = action

        if not self.is_evaluate:
            value = self.critic.forward(state)

            # Collect elements of trajectory
            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.log_probs.append(dist.log_prob(action))
            self.memory.values.append(value)

        # return list(selected_action.detach().cpu().numpy()).pop()
        return selected_action.detach().cpu().numpy().squeeze()

    def _step(self, action: float):
        """
        Make action in environment chosen by current policy,
        if not evaluate - collect elements of trajectory.
        """
        next_state, reward, done, _ = self.env.step(action)

        # Add fake dim to match dimension with batch size
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_evaluate:
            # convert np.ndarray return from environment to torch tensor.
            # collect elements of trajectory.
            self.memory.rewards.append(torch.FloatTensor(reward))
            self.memory.is_terminals.append(torch.FloatTensor(1 - done))

        return next_state, reward, done

    def train(self):
        """
        Interaction process in environment to collect trajectory,
        train process by agent nets after each rollout.
        """
        score = 0
        state = self.env.reset()
        state = np.reshape(state, (1, -1))

        timestep = 0

        for episode in range(self.total_rollouts):
            episode_finished = False
            while not episode_finished:
                action = self._get_action(state)
                next_state, reward, done = self._step(action)

                # Update state and current score
                state = next_state
                current_phi = np.rad2deg(2*np.arccos(state[0][3]))
                score += reward[0][0]

                timestep += 1
                # Update policy after a number of timesteps or after a number of episodes:
                if timestep % self.update_timesteps == 0:
                    value = self.critic.forward(torch.FloatTensor(next_state))
                    self.memory.values.append(value)

                    self._update_weights()

                if done[0][0]:
                    self.scores.append(score)
                    self.avg_score.append(np.mean(self.scores[-100:]))
                    score = 0
                    state = self.env.reset()
                    state = np.reshape(state, (1, -1))
                    episode_finished = True

            print(f'Episode: {episode} \t Score: {self.scores[-1]} \t Avg Score {self.avg_score[-1]}'
                  f'\t Ending PHI {current_phi}')

            # Plot the learning curves after a number of episodes
            # if step_ % self.plot_interval == 0:
            #     self.plot_train_history()

            # If we have achieved the desired score -> stop the process.
            if self.solved_reward is not None:
                if self.avg_score[-1] >= self.solved_reward:
                    print("Congratulations, the environment is solved!")
                    break

            if self.regulate_entropy:
                self.entropy_regularization(self.entropy_start, self.entropy_end)

            if self.save:
                if self.avg_score[-1] >= max(self.avg_score):
                    self.save_model()

        self.env.close()

    def _update_weights(self):
        """Update the model by gradient descent."""

        # print('... updating the networks ...')

        returns = get_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.is_terminals,
            self.gamma,
            self.lamda,
        )
        actor_losses, critic_losses, entropy_vect = [], [], []

        # Flattening a list of torch.tensors into vectors
        states = torch.cat(self.memory.states).view(-1, self.obs_dim)
        actions = torch.cat(self.memory.actions)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.memory.log_probs).detach()
        values = torch.cat(self.memory.values).detach()

        # Compute the advantage
        advantages = returns - values[:-1]

        for state, action, return_, old_log_prob, old_value, advantage in trajectories_data_generator(
                states=states,
                actions=actions,
                returns=returns,
                log_probs=log_probs,
                values=values,
                advantages=advantages,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
        ):
            # Compute ratio (pi_theta / pi_theta__old)
            _, dist = self.actor.forward(state)
            cur_log_prob = dist.log_prob(action)
            ratio = torch.exp(cur_log_prob - old_log_prob)

            # Compute entropy
            entropy = dist.entropy().mean()

            # Compute actor loss
            loss = advantage * ratio
            clipped_loss = (torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantage)
            actor_loss = (
                    -torch.mean(torch.min(loss, clipped_loss))
                    - entropy * self.entropy_coef)

            # Critic loss, uncomment for clipped value loss too.
            cur_value = self.critic.forward(state)
            # clipped_value = (
            #    old_value + torch.clamp(cur_value - old_value,
            #                            -self.value_range, self.value_range)
            #   )
            # loss = (return_ - cur_value).pow(2)
            # clipped_loss = (return_ - clipped_value).pow(2)
            # critic_loss = torch.mean(torch.max(loss, clipped_loss))

            critic_loss = 0.5 * (return_ - cur_value).pow(2).mean()  # Added factor c1 = 0.5

            # Actor optimizer step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic optimizer step
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_vect.append(entropy.item())

        # Clean memory of trajectory
        self.memory.clear_memory()

        # Write mean losses in train history logs: there is a problem when the episode terminates too early!!!
        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        entropy_ = sum(entropy_vect) / len(entropy_vect)
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)
        self.entropy_history.append(entropy_)

    def plot_train_history(self):
        data = [self.scores, self.avg_score, self.actor_loss_history, self.critic_loss_history, self.entropy_history]
        labels = [f"Last episode score: {np.mean(self.scores[-1])}",
                  f"Average score: {self.avg_score[-1]}",
                  f"Actor loss: {np.mean(self.actor_loss_history[-10:])}",
                  f"Critic loss: {np.mean(self.critic_loss_history[-10:])}",
                  f"Entropy: {self.entropy_history[-1]}",
                  ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(len(labels), 1, figsize=(10, 6))
            for i, ax in enumerate(axes[:2]):
                ax.plot(data[i], c="crimson")
                ax.set_title(labels[i])
                ax.set_ylabel('Score')
                ax.set_xlabel('Episode')
                ax.grid(linewidth=0.5)

            for i, ax in enumerate(axes[2:]):
                ax.plot(data[i + 2], c="crimson")
                ax.set_title(labels[i + 2])
                ax.set_ylabel('Score')
                ax.set_xlabel('Update')
                ax.grid(linewidth=0.5)

            plt.tight_layout()
            plt.show()

    def evaluate(self, n_tests=1, plot_q=True):
        print('... testing the model ...')
        self.is_evaluate = True

        for j in range(n_tests):
            state = self.env.reset()
            state = np.reshape(state, (1, -1))
            self.trajectory = state
            done = False
            score = 0

            while not done:
                # self.env.render()
                action = self._get_action(state)
                next_state, reward, done = self._step(action)
                self.trajectory = np.vstack((self.trajectory, next_state))
                # Update state and score
                state = next_state
                state = np.reshape(state, (1, -1))
                score += reward

            self.env.close()

            print(f'Score Test {j}: {np.squeeze(score)} \t '
                  f'Ending PHI {j}: {round(np.rad2deg(2*np.arccos(state[0][3])), 4)}')

        if plot_q:
            # Plot quaternion components evolution:
            data = [self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2],
                    self.trajectory[:, 3], np.rad2deg(2*np.arccos(self.trajectory[:, 3]))]
            labels = [f"q1: {self.trajectory[-1, 0]}",
                      f"q2: {self.trajectory[-1, 1]}",
                      f"q3: {self.trajectory[-1, 2]}",
                      f"q4: {self.trajectory[-1, 3]}",
                      f"PHI: {round(np.rad2deg(2*np.arccos(self.trajectory[-1, 3])), 4)}°"
                      ]

            clear_output(True)
            with plt.style.context("seaborn-dark-palette"):
                fig, axes = plt.subplots(len(labels), 1, figsize=(10, 6))
                for i, ax in enumerate(axes[:]):
                    ax.plot(data[i], c="crimson")
                    ax.set_title(labels[i])
                    ax.set_xlabel('Timestep')
                    ax.grid(linewidth=0.5)

                plt.tight_layout()
                plt.show()

    def entropy_regularization(self, entropy_start, entropy_end):
        # Function for the evolution of the entropy coefficient:
        if self.avg_score[-1] > self.solved_reward / 2:
            self.entropy_coef = entropy_start + ((self.avg_score[-1] - self.solved_reward / 2) / (
                    self.solved_reward - self.solved_reward / 2) * (entropy_end - entropy_start))

    def save_model(self):
        print('... saving models ...')
        torch.save(self.actor.state_dict(), 'Savings/Actors/actor1')
        torch.save(self.critic.state_dict(), 'Savings/Critics/critic1')

    def load_model(self, actor, critic):
        print('... loading models ...')
        self.actor.load_state_dict(torch.load(actor))
        self.critic.load_state_dict(torch.load(critic))

    def SaveArrays(self):

        if self.is_evaluate:
            np.save("Savings/Manim/quaternion.npy", self.trajectory[:, :4])
            np.save("Savings/Manim/omega.npy", self.trajectory[:, 8:])

        else:
            episodes = [i for i in range(len(self.scores))]

            np.save("Savings/Manim/scores.npy", self.scores)
            np.save("Savings/Manim/avg_scores.npy", self.avg_score)
            np.save("Savings/Manim/episodes.npy", episodes)
            np.save("Savings/Manim/actor_loss.npy", self.actor_loss_history)
            np.save("Savings/Manim/critic_loss.npy", self.critic_loss_history)
            np.save("Savings/Manim/entropy.npy", self.entropy_history)

    def Render(self):
        render_simulation(self.trajectory[:, :4])
