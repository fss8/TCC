import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
# import sys
import random
import math
from math import log2





# [[=== BLOCO 1 ===]]
# [[=== _______ ===]]
def energia_voo(DroneF, DroneI):
    Xf, Yf = DroneF # Posição final do drone
    Xi, Yi = DroneI # Posição inicial do drone
    
    # print(DroneF, DroneI)
    Pf = 1 #Potencia de voo
    Vf = 1 #Velocidade de voo
    distancia_voo_quad = math.sqrt((Xf - Xi)**2 + (Yf - Yi)**2)
    Efly = (Pf*distancia_voo_quad)/Vf
    # print(Efly)
    return Efly

#==================================== || ================================
def energia_sobrevoo(DroneI, DispositivoP):
    Xi, Yi = DroneI # Posição inicial do drone
    Ux, Uy = DispositivoP # Posição do usuário
    
    
    Ph = 1 # Potencia de sobrevoo
    Dm = 1 # Numero de bits para computação da tarefa
    Cm = 1 # Número de ciclos de CPU, para computação da tarefa de 1-bit

    # ------ calculo  - para uma tarefa?
    Fm_DT = 1 # Frequência estimada do ciclo de CPU do UAV armazenado no Digital Twin (DT)
    Fm_real_ = 1 # Clico de cpu real
    Fm_desvio = Fm_DT - Fm_real_  # desvio da frequência de CPU entre o seu valor real e o armazenado em DT (avaliar se pode ser 0)

    B = 1 # Banda do sistema
    o2 = 1 # Ruido aditivo Gaussiano branco de potência
    p = 1 # Potencia de transmissão do usuário

    B0 = 1 # ganho de potencia em referencia a distancia de um(1) metro
    distancia_user_modulo_elev_quad = (Ux - Xi)**2 + (Uy - Yi)**2 # distancia até o usuário ||distancia||  elevado ao quadrado
    h = B0/(distancia_user_modulo_elev_quad) # ganho de potencia do canal na LoS(linha de visão)
    RmjUAV = B*log2(1+((p*h)/o2)) # taxa de transmissão do user para o drone (ou do drone para o user?)
    TmjUAV_trans = Dm/RmjUAV

    estimado_Tm = (Dm*Cm)/Fm_DT
    computing_time_gap = (Dm*Cm*Fm_desvio)/(Fm_DT*(Fm_DT-Fm_desvio))
    TmjUAV = estimado_Tm + computing_time_gap
    Ehov = Ph*(TmjUAV_trans + TmjUAV)
    return Ehov, TmjUAV_trans, TmjUAV


# [[=== BLOCO 1 ===]]
# [[=== _______ ===]]
def consumo_joint(DroneF, DroneI, ListaDispositivos):
    Xi, Yi = DroneI # Posição inicial do drone
    Xf, Yf = DroneF # Posição final do drone
    alfa = 1
    beta = 1

    delta = 0.001 # VALOR QUE REPRESENTA O TEMPO ENTRE OS TIMESLOTS / NUMERO DE SLOTS
    teta1 = 1
    teta2 = 1 

    Kuav = 1 # capacitância efetiva

    modulo = math.sqrt((Xf - Xi)**2 + (Yf - Yi)**2)
    velocidade = modulo/delta
    Efly_2 = delta*((teta1*(velocidade**3)) + (teta2/velocidade))

    freqAloc = 1 #por enquanto fixa para qualquer tarefa
    Eexe = 0
    for i in ListaDispositivos:
        Eexe += Kuav*(freqAloc**3)

    Euav = alfa*Eexe + beta*Efly_2
    return Euav

class AgenteRL(gym.Env):
    def __init__(self, max_clients = 25, grid_size = 100, coverage_radius=20):
        super(AgenteRL, self).__init__()
        
        self.coverage_radius = coverage_radius
        self.new_user_prob = 0.04
        self.user_disappearance_prob = 0.04
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

        reward, total_time, reqs, u_waiting = self.get_reward(penalty)

        # self.observation, info = self.get_observation()  #ambiente deveria mudar

       
        
        if reward > 0:
            self.consecutive_positive_rewards += 1
            self.consecutive_negative_rewards = 0
        else:
            self.consecutive_positive_rewards = 0
            self.consecutive_negative_rewards += 1
            
        conditions = self.consecutive_negative_rewards >= self.max_consecutive_negative or self.consecutive_positive_rewards >= self.max_consecutive_positive_rewards
        done = self.is_done(reward) or conditions
        info = {"tempos": self.total_time_waiting, "qtdw": self.num_waiting , "accepts": self.sum_accepts, "rejects": self.sum_rejects}

        self.state = self._get_state()
        return self.state, reward, done, {}, info
    
    def undeterministic_random_movement(self):
        #Geração de um novo usuário
        if np.random.rand() < self.new_user_prob:
            # Sorteia um lugar aleatório no grid
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                if self.clientes_grid[pos[0]][pos[1]] == 0:
                    break
            new_user_position = pos.tolist()
            self.users_positions.append(new_user_position)
            self.users_time.append(0)
            self.user_states = np.append(self.user_states, 1)  # 1 para usuário novo
            self.clientes_grid[new_user_position[0]][new_user_position[1]] = 1
            
        index = 0
        while index < self.user_states.shape[0]:
            if self.user_states[index] == 1:
                if np.random.rand() < self.user_disappearance_prob/10:
                    user_index = index
                    
                    self.clientes_grid[self.users_positions[user_index][0]][self.users_positions[user_index][1]] = 0
                    self.users_positions.pop(user_index)
                    self.users_time.pop(user_index)
                    self.user_states = np.delete(self.user_states,user_index, 0)
                else:
                    index += 1
            else:
                index += 1

            
        #Mudanças de estado
        for index, user in enumerate(self.user_states):
            if user == 1:
                if np.random.rand() < 0.01:
                    self.clientes_grid[self.users_positions[index][0]][self.users_positions[index][1]] = 2
                    self.user_states[index] = 2
                    self.users_time[index] += 1
                    
            elif user == 3: # Usuário se moveu
                pass
                # if np.random.rand() < 0.5: # 50% de chance de mover
                #     # Sorteia um lugar aleatório no grid
                #     while True:
                #         pos = np.random.randint(0, self.grid_size, size=2)
                #         if self.clientes_grid[pos[0]][pos[1]] == 0:
                #             break
                #     new_user_position = pos
                #     self.users_positions[index] = new_user_position
                #     self.clientes_grid[new_user_position[0]][new_user_position[1]] = 1
                #     self.user_states[index] = 0
        
        
    
    def take_action(self, action):
        last_position = self.posicao.copy()
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
        energy_COST = energia_voo(self.posicao, last_position)
        # self.battery -= speed/10
        energy_penalty = energy_COST
        self.battery -= energy_COST
        return self.posicao, energy_penalty
    
    def get_reward(self, energy_penalty):
        reward = 0
        users_waiting = 0
        sum_time = 0
        qty = 0
        # media = np.mean(self.users_time)
        energy_sobrevoo = 0
        for i in range(len(self.users_positions)):
        
            if self.user_states[i] == 3:
                tempo_esperando = self.users_time[i]
                if np.linalg.norm(self.posicao - self.users_positions[i]) <= self.coverage_radius:
                    reward += 500/(10+tempo_esperando)  # Adiciona 1 para evitar divisão por zero
                    self.sum_accepts += 1
                    # sum_time += self.users_time[i]
                    # users_waiting += 1
                else: # Penalidade por ter usuário sem cobertura
                    reward -= 2*tempo_esperando
                    self.sum_rejects += 1
                self.user_states[i] = 1
                self.users_time[i] = 0
            elif self.user_states[i] == 2:
                if np.linalg.norm(self.posicao - self.users_positions[i]) <= self.coverage_radius:
                    self.user_states[i] = 3
                    sum_time += self.users_time[i]
                    # self.users_time[i] +=1
                    users_waiting += 1
                    qty += 1
                    reward = 10
                    
                    cost_voo, _, _= energia_sobrevoo(self.posicao, self.users_positions[i])
                    energy_sobrevoo -= cost_voo
                    
                else:
                    self.users_time[i] += 1
                    reward -= self.users_time[i]/10
        energy_penalty = energy_penalty + energy_sobrevoo
        if reward > 0:
            reward = reward - (energy_penalty/12)
        else:
            reward = (reward*(energy_penalty/10))/(len(self.user_states)+1)
        # print(reward)
        # reward = reward - (users_waiting)
        self.total_time_waiting += sum_time
        self.num_waiting += users_waiting
        return reward, sum_time, qty, users_waiting
    
    # def get_observation(self):
    #     return self.observation, {}
    
    def is_done(self, reward):
        return self.battery <= 0 #or np.all(self.user_states == 2)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.battery = self.max_battery_inital
        self.consecutive_positive_rewards = 0 
        self.consecutive_negative_rewards = 0
        self.max_consecutive_positive_rewards = 50
        self.max_consecutive_negative = 250

        self.sum_accepts = 0
        self.sum_rejects = 0
        self.total_time_waiting = 0
        self.num_waiting = 0
        
        # self.posicao = np.random.integers(0, self.grid_size, size=2, dtype=int)
        self.posicao = np.random.randint(0, self.grid_size, size=2)
        self.clientes_grid = np.zeros((self.grid_size, self.grid_size), dtype=int) #deveria gerar aleatóriamente a posição dos (tambem aleatoriamente quantidade de )usuários
        self.users_positions = []
        self.users_time = []
        # self.user_states = [ ]
        
        random_clients_qty = np.random.randint(1, self.maxclients)
        self.user_states = np.zeros(random_clients_qty)
        for i in range(random_clients_qty):
            #verificar se possui já um cliente ali
            
            user_position = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            self.clientes_grid[user_position[0]][user_position[1]] = 1
            self.user_states[i] = 1
            self.users_time.append(0)
            self.users_positions.append(user_position)
            
        self.state = self._get_state()
        return self.state, ''
    
    def render(self, screen, episode, total_reward, step):
        
        screen.fill((255, 255, 255))  # Limpa a tela com branco
        block_size = 8  # Tamanho de cada bloco no grid
        color = (0,0, 255)
        for i in range(0, 100):
            for y in range(0,100):
                if np.linalg.norm(self.posicao - (i, y)) == self.coverage_radius:
                    pygame.draw.rect(screen, color, (y * block_size, i * block_size, block_size, block_size))
        # Desenha os usuários
        for pos, state in zip(self.users_positions, self.user_states):
            if state == 1:
                color = (0, 10, 0)  # Preto se disponível
            elif state == 3:
                # Define a cor com base na cobertura do drone
                if np.linalg.norm(self.posicao - pos) <= self.coverage_radius:
                    color = (0, 0, 255)  # Azul se dentro da área de cobertura
                else:
                    color = (255, 0, 0)  # Vermelho se fora da área de cobertura
            else:
                color = (0, 255, 0)  # Outro estadooo &&****&*¨&*&*&*
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