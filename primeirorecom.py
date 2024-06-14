import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import pygame
import sys

class DroneEnv(gym.Env):
    def __init__(self, grid_size=10, num_users=5, user_movement_prob=0.5, coverage_radius=10):
        super(DroneEnv, self).__init__()
        self.grid_size = grid_size
        self.num_users = num_users
        self.user_movement_prob = user_movement_prob
        self.coverage_radius = coverage_radius
        
        self.consecutive_positive_rewards = 0  # Contador de recompensas positivas consecutivas
        self.max_consecutive_positive_rewards = 10  # Limite para considerar a parada
        self.action_space = gym.spaces.Discrete(5)  # 4 movimentos: norte, sul, leste, oeste e PARADO
        self.observation_space = gym.spaces.Box(0, grid_size, shape=(2 + num_users * 3,), dtype=np.float32)  # 2 para posição do drone, 3*num_users para posições e estados dos usuários
        self.reset()

    def reset(self):
        self.drone_position = np.random.randint(0, self.grid_size, size=2)
        self.user_positions = np.random.randint(0, self.grid_size, size=(self.num_users, 2))
        self.user_states = np.zeros(self.num_users)
        self.state = np.concatenate([self.drone_position, self.user_positions.flatten(), self.user_states])
        return self.state

    def step(self, action):
        prev_drone_position = self.drone_position.copy()  # Salva a posição anterior do drone

        if action == 0:
            self.drone_position[0] = max(0, self.drone_position[0] - 1)  # norte
        elif action == 1:
            self.drone_position[0] = min(self.grid_size - 1, self.drone_position[0] + 1)  # sul
        elif action == 2:
            self.drone_position[1] = min(self.grid_size - 1, self.drone_position[1] + 1)  # leste
        elif action == 3:
            self.drone_position[1] = max(0, self.drone_position[1] - 1)  # oeste
        elif action == 4:
            pass  # Ficar parado

        reward = 0
        

        self.state = np.concatenate([self.drone_position, self.user_positions.flatten(), self.user_states])
        done = np.all(self.user_states == 1)
        for i in range(self.num_users):
            # Usuário pode se mover com uma certa probabilidade
            # if np.random.rand() < self.user_movement_prob:
            #     self.user_positions[i] = self.user_positions[i] + np.random.randint(-1, 2, size=2)
            #     self.user_positions[i] = np.clip(self.user_positions[i], 0, self.grid_size - 1)

            if np.linalg.norm(self.drone_position - self.user_positions[i]) <= self.coverage_radius:
                self.user_states[i] = 1
                distance = np.linalg.norm(self.drone_position - self.user_positions[i])
                reward += 15 / (distance + 1)  # Adiciona 1 para evitar divisão por zero
            elif self.user_states[i] == 1:
                self.user_states[i] = 0
                reward -= 1
            else: # Penalidade por ter usuário sem cobertura
                reward -= 1
                
        total_reward = reward
                
        # Verifica se a recompensa é positiva
        if total_reward > 0:
            self.consecutive_positive_rewards += 1
        else:
            self.consecutive_positive_rewards = 0

        # Verifica se o critério de parada foi atingido
        if self.consecutive_positive_rewards >= self.max_consecutive_positive_rewards:
            print(self.consecutive_positive_rewards, self.max_consecutive_positive_rewards)
            done = True
        else:
            done = np.all(self.user_states == 1)

        self.state = np.concatenate([self.drone_position, self.user_positions.flatten(), self.user_states])
        # done = np.all(self.user_states == 1)
        return self.state, reward, done, {}

    def render(self, screen, episode, total_reward, step):
        screen.fill((255, 255, 255))  # Limpa a tela com branco
        block_size = 40  # Tamanho de cada bloco no grid

        # Desenha os usuários
        for pos, state in zip(self.user_positions, self.user_states):
            if state == 1:
                color = (0, 255, 0)  # Verde se atendido
            else:
                # Define a cor com base na cobertura do drone
                if np.linalg.norm(self.drone_position - pos) <= self.coverage_radius:
                    color = (0, 0, 255)  # Azul se dentro da área de cobertura
                else:
                    color = (255, 0, 0)  # Vermelho se fora da área de cobertura
            pygame.draw.rect(screen, color, (pos[1] * block_size, pos[0] * block_size, block_size, block_size))

        # Desenha o drone
        pygame.draw.rect(screen, (0, 0, 255), (self.drone_position[1] * block_size, self.drone_position[0] * block_size, block_size, block_size))

        # Adiciona texto de status
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Episode: {episode} | Step: {step} | Total Reward: {total_reward}', True, (0, 0, 0))
        screen.blit(text, (10, self.grid_size * block_size + 10))

        pygame.display.flip()  # Atualiza a tela

def build_model(state_shape, num_actions):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=state_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_dqn(env, num_episodes=1000):
    state_shape = (env.observation_space.shape[0],)
    model = build_model(state_shape, env.action_space.n)
    target_model = build_model(state_shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.95

    replay_buffer = []
    batch_size = 12

    pygame.init()
    screen = pygame.display.set_mode((env.grid_size * 40, env.grid_size * 40 + 50))  # Tamanho da tela do Pygame
    pygame.display.set_caption('Drone Environment')

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, -1])
        total_reward = 0

        for t in range(200):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = env.step(action)
            # print(done)
            next_state = np.reshape(next_state, [1, -1])
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Renderiza o ambiente
            env.render(screen, episode, total_reward, t)
            
            if done:
                print(done)
                break

            if len(replay_buffer) > batch_size:
                minibatch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                # print(minibatch)
                for i in minibatch:
                    s, a, r, s_next, d = replay_buffer[i]
                    target = r
                    if not d:
                        target += gamma * np.amax(target_model.predict(s_next, verbose=0)[0])
                    target_q_values = model.predict(s, verbose=0)
                    target_q_values[0][a] = target
                    model.train_on_batch(s, target_q_values)

                target_model.set_weights(model.get_weights())

            

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    pygame.quit()

    return model

env = DroneEnv(grid_size=20, num_users=10)
model = train_dqn(env)