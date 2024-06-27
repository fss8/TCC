import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
# import sys
import random

class AgenteRL(gym.Env):
    def __init__(self, max_clients = 25, grid_size = 100, coverage_radius=12):
        super(AgenteRL, self).__init__()
        
        self.coverage_radius = coverage_radius
        self.new_user_prob = 0.1
        self.user_disappearance_prob = 0.1
        self.maxclients = max_clients
        self.grid_size = grid_size
        # self.action_space = spaces.Discrete(5)
        self.max_battery_inital = 1000
        self.battery = self.max_battery_inital
        self.action_space = gym.spaces.Discrete(21)  
        self.observation_space = spaces.Box(low=0, high=1000, shape=(grid_size*grid_size + 2+1,), dtype=int)
        # spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.float32)  #Imagem com 3 cores rgb(transformar em uma matriz com 0 ou 1, contendo as posições q possuem
        
        self.reset()
        
    def _get_state(self):
        return np.concatenate([self.clientes_grid.flatten(), self.posicao, [self.battery]])

    def step(self, action):
        
        self.undeterministic_random_movement()
        
        acao, penalty = self.take_action(action)

        reward = self.get_reward(penalty)

        # self.observation, info = self.get_observation()  #ambiente deveria mudar

       
        
        if reward > 0:
            self.consecutive_positive_rewards += 1
            self.consecutive_negative_rewards = 0
        else:
            self.consecutive_positive_rewards = 0
            self.consecutive_negative_rewards += 1
            
        conditions = self.consecutive_negative_rewards >= self.max_consecutive_negative or self.consecutive_positive_rewards >= self.max_consecutive_positive_rewards
        done = self.is_done(reward) or conditions
        info = {}

        self.state = self._get_state()
        return self.state, reward, done, {}, info
    
    def undeterministic_random_movement(self):
        if np.random.rand() < self.new_user_prob:
            # Sorteia um lugar aleatório no grid
            new_user_position = np.random.randint(0, self.grid_size, size=2)
            self.users_positions.append(new_user_position)
            self.user_states = np.append(self.user_states, 0)  # 0 para usuário sem cobertura
            self.clientes_grid[new_user_position[0]][new_user_position[1]] = 1
            
        if np.random.rand() < self.user_disappearance_prob:
            user_index = np.random.randint(len(self.users_positions))
            
            self.clientes_grid[self.users_positions[user_index][0]][self.users_positions[user_index][1]] = 0
            self.users_positions.pop(user_index)
            self.user_states = np.delete(self.user_states,user_index, 0)
        
        
    
    def take_action(self, action):
        direction = action // 5
        speed = (action % 5) + 1  # Speed range from 1 to 5
        if action == 20: # parado
            speed = 0
        elif direction == 0:
            self.posicao[0] = max(0, self.posicao[0] - speed)  # norte
        elif direction == 1:
            self.posicao[0] = min(self.grid_size - 1, self.posicao[0] + speed)  # sul
        elif direction == 2:
            self.posicao[1] = min(self.grid_size - 1, self.posicao[1] + speed)  # leste
        elif direction == 3:
            self.posicao[1] = max(0, self.posicao[1] - speed)  # oeste
        # elif action == 4:
            # pass  # Ficar parado
        self.battery -= speed/10
        energy_penalty = speed * 2
        return self.posicao, energy_penalty
    
    def get_reward(self, energy_penalty):
        reward = 0
        for i in range(len(self.users_positions)):
            if np.linalg.norm(self.posicao - self.users_positions[i]) <= self.coverage_radius:
                # self.user_states[i] = 1
                reward += 20  # Adiciona 1 para evitar divisão por zero
                if self.user_states[i] == 0:
                    self.user_states[i] = 1
            else: # Penalidade por ter usuário sem cobertura
                self.user_states[i] = 0
                # self.clientes_grid[self.users_positions[i][0]][self.users_positions[i][1]] = 0
                reward -= 1
        reward = reward - energy_penalty
        return reward
    
    # def get_observation(self):
    #     return self.observation, {}
    
    def is_done(self, reward):
        return self.battery <= 0 or np.all(self.user_states == 1)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.battery = self.max_battery_inital
        self.consecutive_positive_rewards = 0 
        self.consecutive_negative_rewards = 0
        self.max_consecutive_positive_rewards = 40
        self.max_consecutive_negative = 60
        
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
        block_size = 8  # Tamanho de cada bloco no grid

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
        text = font.render(f'Episode: {episode} | Step: {step} | Total Reward: {total_reward} | battery: {self.battery}', True, (0, 0, 0))
        screen.blit(text, (10, self.grid_size * block_size + 10))

        pygame.display.flip()  # Atualiza a tela
        

    def play_step(self):
        pass
        
    def seed(self, seed=None):
        pass

    def close(self):
        pass