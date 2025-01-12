import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
# from opt_env_hiv import FastHIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters
params = {
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY": 0.99,
    "TARGET_UPDATE_FREQ": 400,
    "MEMORY_SIZE": 500000,
    "BATCH_SIZE": 1000,
    "NUM_NEURONS": 256,
    "NUM_LAYERS": 5,
    "LEARNING_RATE": 1e-3,
    "MULTIPLE_GRAD_STEPS": 3,
    "USE_SCHEDULER": False,
    "CRITERION": nn.MSELoss(),
}

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.d = params["NUM_NEURONS"]
        self.num_layers = params["NUM_LAYERS"]
        self.fc1 = nn.Linear(state_size, self.d)
        self.dense_layers = nn.ModuleList([nn.Linear(self.d, self.d) for _ in range(self.num_layers)])
        self.fc4 = nn.Linear(self.d, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for layer in self.dense_layers:
            x = torch.relu(layer(x))
        return self.fc4(x)

# DQN Agent Implementation
class ProjectAgent:
    def __init__(self):
        self.state_size = 6
        self.action_size = 4
        self.epsilon = params["EPSILON_START"]
        self.memory = deque(maxlen=params["MEMORY_SIZE"])
        self.gamma = params["GAMMA"]

        # Networks
        self.policy_net = DQNetwork(self.state_size, self.action_size).float()
        self.target_net = DQNetwork(self.state_size, self.action_size).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params["LEARNING_RATE"])
        self.criterion = params["CRITERION"]
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)

    def act(self, state, use_random=False):
        """
        Select an action based on the observation and a greedy policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Perform experience replay to train the policy network.
        """
        if len(self.memory) < params["BATCH_SIZE"]:
            return

        for _ in range(params["MULTIPLE_GRAD_STEPS"]):
            # Sample a batch of experiences
            batch = random.sample(self.memory, params["BATCH_SIZE"])
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            QYmax = self.target_net(next_states).max(1)[0].detach()
            update = torch.addcmul(rewards, 1-dones, QYmax, value=self.gamma)
            QXA = self.policy_net(states).gather(1, actions.unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            if params["USE_SCHEDULER"]:
                self.scheduler.step()

    def update_target_network(self):
        """
        Update the target network with the policy network weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """
        Save the policy network's weights.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self):
        """
        Load the policy network's weights.
        """
        model_type="best"
        if os.path.exists("./" + model_type + "_model.pth"):
            self.policy_net.load_state_dict(torch.load("./" + model_type + "_model.pth", map_location=torch.device("cpu")))
            self.policy_net.eval()
            print(f"Model loaded from ./" + model_type + "_model.pth")
        else:
            print(f"No model found at ./" + model_type + "_model.pth")

    def train(self, num_episodes=300, max_timesteps=200, save_path="./final_model.pth"):
        """
        Train the agent on the environment.
        """
        print(f"Training for {num_episodes} episodes...")

        seed_everything()

        train_rewards = []
        val_rewards_fixed = []
        val_rewards_rand = []
        step = 0
        best_val_reward = 0
        env = HIVPatient(domain_randomization=False)
        env_rand = HIVPatient(domain_randomization=True)
        for episode in range(num_episodes):
            total_reward = 0
            if episode%6 == 0:
                cur_env = env_rand
            else:
                cur_env = env
            state, _ = cur_env.reset()

            for t in range(max_timesteps):
                step += 1
                if np.random.rand() < self.epsilon:
                    action = cur_env.action_space.sample()
                else:
                    action = self.act(state, use_random=False)
                next_state, reward, done, _, _ = cur_env.step(action)
                self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                self.replay()
                if done:
                    break

            # Update target network periodically
            if step % params["TARGET_UPDATE_FREQ"] == 0:
                self.update_target_network()

            # Decay epsilon
            self.epsilon = max(params["EPSILON_END"], self.epsilon * params["EPSILON_DECAY"])

            if episode % 5 == 0:
                val_reward_fixed = evaluate_HIV(self, nb_episode=1)
                val_reward_rand = evaluate_HIV_population(self, nb_episode=10)

                val_reward = min(val_reward_fixed, val_reward_rand)

                # Logging
                print(f"Episode {episode}/{num_episodes} | "
                    f"Episode Reward: {total_reward:.2e} | "
                    f"Validation Reward Fixed: {val_reward_fixed:.2e} | "
                    f"Validation Reward Random: {val_reward_rand:.2e} | "
                    f"Epsilon: {self.epsilon:.2f}")
                
                val_rewards_fixed.append(val_reward_fixed)
                val_rewards_rand.append(val_reward_rand)
                train_rewards.append(total_reward)

                # Save the best model
                if val_reward >= best_val_reward:
                    best_val_reward = val_reward
                    self.save("./best_model.pth")

        # Save final model
        self.save(save_path)

        # Plot rewards
        plot_rewards(train_rewards, val_rewards_fixed, val_rewards_rand)

def plot_rewards(train_rewards, val_rewards_fixed, val_rewards_rand):
    """
    Plot training rewards.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label="Training Reward")
    plt.plot(val_rewards_fixed, label="Validation Reward Fixed")
    plt.plot(val_rewards_rand, label="Validation Reward Random")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.yscale("log")
    plt.title("Training and Validation Rewards")
    plt.legend()
    plt.grid()
    # create logs directory if it does not exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    plt.savefig("./logs/rewards_plot.png")
    plt.show()