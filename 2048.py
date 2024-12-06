import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy

# ============= Implementação do Jogo 2048 ==================

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self._add_new_tile()
        self._add_new_tile()

    def _add_new_tile(self):
        empty_positions = list(zip(*np.where(self.board == 0)))
        if not empty_positions:
            return
        x, y = random.choice(empty_positions)
        self.board[x, y] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        return self._move(self.board, axis=1, reverse=False)

    def move_right(self):
        return self._move(self.board, axis=1, reverse=True)

    def move_up(self):
        return self._move(self.board, axis=0, reverse=False)

    def move_down(self):
        return self._move(self.board, axis=0, reverse=True)

    def _move(self, board, axis, reverse):
        moved = False
        score_gain = 0
        new_board = np.zeros_like(board)

        for i in range(self.size):
            line = board[i, :] if axis == 0 else board[:, i]
            if reverse:
                line = line[::-1]

            compacted_line = line[line != 0]
            combined_line = []
            skip = False
            for j in range(len(compacted_line)):
                if skip:
                    skip = False
                    continue
                if j < len(compacted_line)-1 and compacted_line[j] == compacted_line[j+1]:
                    new_val = compacted_line[j]*2
                    combined_line.append(new_val)
                    score_gain += new_val
                    skip = True
                else:
                    combined_line.append(compacted_line[j])

            while len(combined_line) < self.size:
                combined_line.append(0)

            if reverse:
                combined_line = combined_line[::-1]

            if axis == 0:
                new_board[i, :] = combined_line
            else:
                new_board[:, i] = combined_line

            # Verifica se mudou a linha
            if not np.array_equal(line if not reverse else line[::-1], combined_line):
                moved = True

        if moved:
            self.board = new_board
            self.score += score_gain
            self._add_new_tile()
        return moved, score_gain

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for x in range(self.size):
            for y in range(self.size):
                val = self.board[x,y]
                if x+1 < self.size and self.board[x+1,y] == val:
                    return False
                if y+1 < self.size and self.board[x,y+1] == val:
                    return False
        return True

    def get_state(self):
        # Convertendo os valores em log2 para estabilizar
        # log2(0) não é definido, então usamos log2(valor) se valor>0 senão 0
        state = []
        for val in self.board.flatten():
            if val > 0:
                state.append(math.log2(val))
            else:
                state.append(0)
        return np.array(state, dtype=np.float32)

    def get_score(self):
        return self.score


# ============= Rede Neural (DQN) ==================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============= Agente DQN ==================

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, lr=0.001, batch_size=64, memory_size=10000, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01, target_update=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.learn_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            state_t = torch.tensor([state], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.bool).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # Q_target
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + (self.gamma * max_next_q * (~dones))

        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============= Execução do Treino ==================

def play_game(env, agent, actions_map):
    state = env.get_state()
    total_score = 0
    done = False
    while not env.is_game_over():
        action_idx = agent.act(state)
        action_name = actions_map[action_idx]

        moved = False
        reward = 0
        if action_name == 'up':
            moved, r = env.move_up()
        elif action_name == 'down':
            moved, r = env.move_down()
        elif action_name == 'left':
            moved, r = env.move_left()
        elif action_name == 'right':
            moved, r = env.move_right()

        if moved:
            next_state = env.get_state()
            reward = r
            total_score = env.get_score()
            done = env.is_game_over()
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
        else:
            # Jogada inválida - podemos dar uma pequena punição
            # para desencorajar ações que não alteram o estado
            agent.remember(state, action_idx, -1, state, False)

        if done:
            break
    return total_score


if __name__ == "__main__":
    actions = ['up', 'down', 'left', 'right']
    state_size = 16  # 4x4
    action_size = len(actions)

    agent = DQNAgent(state_size, action_size, gamma=0.9, lr=0.001, batch_size=64, memory_size=20000, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, target_update=500)

    episodes = 5000
    scores = []
    for e in range(episodes):
        env = Game2048(size=4)
        score = play_game(env, agent, actions)
        agent.replay()
        scores.append(score)

        if (e+1) % 500 == 0:
            avg_score = np.mean(scores[-500:])
            print(f"Episode: {e+1}, Avg Score (last 500): {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Ao final, pode-se observar se a pontuação média está aumentando.
