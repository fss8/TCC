from collections import deque
from turtle import Screen
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
from gymnasium import spaces
import os
import pygame
import sys
import random

import time as tm

from tensorflow.keras.models import load_model
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

class AgenteRL(gym.Env):
    def __init__(self, max_clients = 20, grid_size = 30, coverage_radius=12):
        super(AgenteRL, self).__init__()
        
        self.coverage_radius = coverage_radius
        self.maxclients = max_clients
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(grid_size*grid_size + 2,), dtype=int)
        # spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.float32)  #Imagem com 3 cores rgb(transformar em uma matriz com 0 ou 1, contendo as posições q possuem
        
        # self.reset()
        
    def _get_state(self):
        return np.concatenate([self.posicao, self.clientes_grid.flatten()]).reshape(1, -1)

    def step(self, action):
        
        self.take_action(action)

        reward = self.get_reward()

        # self.observation, info = self.get_observation()  #ambiente deveria mudar

       
        
        if reward > 0:
            self.consecutive_positive_rewards += 1
        else:
            self.consecutive_positive_rewards = 0
            
        done = self.is_done(reward) or self.consecutive_positive_rewards >= self.max_consecutive_positive_rewards
        info = {}

        self.state = self._get_state()
        return self.state, reward, done, {}, info
    
    def take_action(self, action):
        if action == 0:
            self.posicao[0] = max(0, self.posicao[0] - 1)  # norte
        elif action == 1:
            self.posicao[0] = min(self.grid_size - 1, self.posicao[0] + 1)  # sul
        elif action == 2:
            self.posicao[1] = min(self.grid_size - 1, self.posicao[1] + 1)  # leste
        elif action == 3:
            self.posicao[1] = max(0, self.posicao[1] - 1)  # oeste
        elif action == 4:
            pass  # Ficar parado
        return self.posicao
    
    def get_reward(self):
        reward = 0
        for i in range(len(self.users_positions)):
            # Usuário pode se mover com uma certa probabilidade
            # if np.random.rand() < self.user_movement_prob:
            #     self.user_positions[i] = self.user_positions[i] + np.random.randint(-1, 2, size=2)
            #     self.user_positions[i] = np.clip(self.user_positions[i], 0, self.grid_size - 1)

            # distance = np.linalg.norm(self.posicao - self.users_positions[i])
            if np.linalg.norm(self.posicao - self.users_positions[i]) <= self.coverage_radius:
                # self.user_states[i] = 1
                reward += 3  # Adiciona 1 para evitar divisão por zero
                if self.user_states[i] == 0:
                    self.user_states[i] = 1
                    self.clientes_grid[self.users_positions[i][0]][self.users_positions[i][1]] = 1
                    # reward += 5
                    # reward += 10
                # self.user_states[i] = 1
            # elif self.user_states[i] == 1:
            #     self.user_states[i] = 0
            #     reward -= 1
            else: # Penalidade por ter usuário sem cobertura
                self.user_states[i] = 0
                self.clientes_grid[self.users_positions[i][0]][self.users_positions[i][1]] = 0
                reward -= 1
        return reward
    
    # def get_observation(self):
    #     return self.observation, {}
    
    def is_done(self, reward):
        # if reward >= 20:
        #     return True
        # if self.posicao == self.observation["target"]:
        #     return True
        # if self.observation["steps"] >= 100:
        #     return True
        # for e in self.users_positions:
        #     if (e == self.posicao).all():
        #         return True
        return np.all(self.user_states == 1)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.consecutive_positive_rewards = 0 
        self.max_consecutive_positive_rewards = 40
        
        # self.posicao = np.random.integers(0, self.grid_size, size=2, dtype=int)
        self.posicao = np.random.randint(0, self.grid_size, size=2)
        self.clientes_grid = np.zeros((self.grid_size, self.grid_size), dtype=int) #deveria gerar aleatóriamente a posição dos (tambem aleatoriamente quantidade de )usuários
        self.users_positions = []
        # self.user_states = [ ]
        
        random_clients_qty = np.random.randint(1, self.maxclients)
        self.user_states = np.zeros(random_clients_qty)
        for i in range(random_clients_qty):
            #verificar se possui já um cliente ali
            
            user_position = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            self.clientes_grid[user_position[0]][user_position[1]] = 1
            self.users_positions.append(user_position)
            
        self.state = self._get_state()
        return self.state, ''
    
    def render(self, screen, episode, total_reward, step):
        
        screen.fill((255, 255, 255))  # Limpa a tela com branco
        block_size = 20  # Tamanho de cada bloco no grid

        # Desenha os usuários
        for pos, state in zip(self.users_positions, self.user_states):
            if state == 1:
                color = (0, 255, 0)  # Verde se atendido
            else:
                # Define a cor com base na cobertura do drone
                if np.linalg.norm(self.posicao - pos) <= self.coverage_radius:
                    color = (0, 0, 255)  # Azul se dentro da área de cobertura
                else:
                    color = (255, 0, 0)  # Vermelho se fora da área de cobertura
            pygame.draw.rect(screen, color, (pos[1] * block_size, pos[0] * block_size, block_size, block_size))

        # Desenha o drone
        pygame.draw.rect(screen, (0, 0, 255), (self.posicao[1] * block_size, self.posicao[0] * block_size, block_size, block_size))

        # Adiciona texto de status
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Episode: {episode} | Step: {step} | Total Reward: {total_reward}', True, (0, 0, 0))
        screen.blit(text, (10, self.grid_size * block_size + 10))

        pygame.display.flip()  # Atualiza a tela
        

    def play(self):
        pass
        
    def seed(self, seed=None):
        pass

    def close(self):
        pass
        # pygame.quit()
        
# def build_model(env, learning_rate):
#     arr1 = env.observation_space['clientes-grid'].sample().flatten()
#     print(arr1)
#     arr2 = env.observation_space['posicao'].sample().flatten()
#     print(arr2)
#     input_model = (arr1.shape[0] + arr2.shape[0],)
#     print("INPUT MODELLLL:", input_model)
#     num_actions = env.action_space.n
    
#     # input_

#     dqn = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape = input_model),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(num_actions, activation='linear')
#     ])
#     dqn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
#                 loss=tf.keras.losses.Huber())

#     return dqn

class Callback(tf.keras.callbacks.Callback):
    def on_episode_end(self, episode, logs):
        if episode % 100 == 0:
            self.model.save('dqn_model.h5')

class SuppressOutput(tf.keras.callbacks.Callback):
    def __enter__(self):
        self._original_stdout = os.dup(1)
        self._original_stderr = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._original_stdout, 1)
        os.dup2(self._original_stderr, 2)
        os.close(self._devnull)

    def on_epoch_end(self, epoch, logs=None):
        pass  # This can be customized to print something at the end of an epoch if needed

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.92
        self.learning_rate = 0.001
        self.suppress_output = SuppressOutput()
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon: # EXPLORATION VS EXPLOITATION
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + Q_future * self.gamma
            # print(target)
            # print(target[0])
            
            # self.model.set_weights(self.model.get_weights())
            self.model.fit(state, target, epochs=1 , verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save(name)

  
def train_dqn(env, num_episodes):
    # Configurações do DQN
    
    # dqn = build_model(env, learning_rate=0.001)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # episodes = 1000
    batch_size = 15
    
    # dqn.set_weights(dqn.get_weights()) #talvez seja inútil
    
    
    total_reward = 0
    
    # pygame.init()
    # screen = pygame.display.set_mode((env.grid_size * 20, env.grid_size * 20 + 30))  # Tamanho da tela do Pygame
    # pygame.display.set_caption('Drone Environment')

    # Treinamento do DQN
    for episode in range(num_episodes):
        state = env.reset()
        # print(state)
        total_reward = 0
        
        
        done = False
        for time in range(20):
            # env.render(screen, episode, total_reward, time)
            # time.sleep(0.1)
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # print(reward)
            
            reward = reward if not done else 50 # A VERIFICAR
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {episode}, score: {total_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # flat_state = np.concatenate((state["clientes-grid"].flatten(), state["posicao"])).reshape(1, -1)
            # new_flat_state = np.concatenate((next_state["clientes-grid"].flatten(), next_state["posicao"]))
        print("TOTAL REWARD:", total_reward)
        
        agent.save(f"modelos/dqn{episode}.h5")
        
    # pygame.quit()

    agent.save("dqn.h5")
    return agent


# Função para testar o modelo carregado
def test_agent(env, model, episodes=10):
    
    pygame.init()
    screen = pygame.display.set_mode((env.grid_size * 20, env.grid_size * 20 + 30))  # Tamanho da tela do Pygame
    pygame.display.set_caption('Drone Environment')
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        for time in range(300):
            env.render(screen, e, total_reward, time)
            
            action = np.argmax(model.predict(state, verbose=0))
            # print(action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}")
                break
    pygame.quit()
    
def test_ppo(env, model, episodes=10):
    pygame.init()
    screen = pygame.display.set_mode((env.grid_size * 20, env.grid_size * 20 + 30))  # Tamanho da tela do Pygame
    pygame.display.set_caption('Drone Environment')
    for e in range(episodes):
        state, _ = env.reset()
        # print(state)
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        for time in range(500):
            tm.sleep(0.1)
            env.render(screen, e, total_reward, time)
            action, _ = model.predict(state)
            # print(action)
            state, reward, done, _, _ = env.step(action[0])
            total_reward += reward
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}")
                break
    # pygame.quit()


if __name__ == '__main__':
    env = AgenteRL()
    
    
    # Modelo PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=80000)
    model.save("ppo_drone_model3")
    model = PPO.load("ppo_drone_model3")
    
    test_ppo(env, model)
    
    
    # model = load_model("modelos/dqn19.h5")
    # test_agent(env, model)
    # env = DroneEnv(grid_size=20, num_users=10)
    # model = train_dqn(env, num_episodes=20)