import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
# import sys
import random
import math
from math import log2

import time



# [[=== BLOCO 1 ===]]
# [[=== _______ ===]]
def energia_voo(DroneF, DroneI, velocity):
    Xf, Yf = DroneF # Posição final do drone
    Xi, Yi = DroneI # Posição inicial do drone
    # print(velocity)
    # print(DroneF, DroneI)
    # print(DroneF, DroneI)
    if(velocity > 0):
        Pf = 200 #Potencia de voo (em watts?)
        Vf = 250 #Velocidade de voo
        # print(DroneF, DroneI)
        distancia_voo_quad = math.sqrt((Xf - Xi)**2 + (Yf - Yi)**2)
        # print(distancia_voo_quad)
        Efly = (Pf*distancia_voo_quad)/Vf
    else: Efly = 0
    # print(Efly)
    return Efly

#==================================== || ================================
def energia_sobrevoo(DroneI, DispositivoP):
    Xi, Yi = DroneI # Posição inicial do drone
    Ux, Uy = DispositivoP # Posição do usuário
    
    
    Ph = 200 # Potencia de sobrevoo
    Dm = 8_000_000 # Numero de bits para computação da tarefa
    Cm = 1000 # Número de ciclos de CPU, para computação da tarefa de 1-bit

    # ------ calculo  - para uma tarefa?
    Fm_DT = 2_000_000_000 # Frequência estimada do ciclo de CPU do UAV armazenado no Digital Twin (DT)
    Fm_real_ = 2_000_000_000 # Clico de cpu real
    Fm_desvio = Fm_DT - Fm_real_  # desvio da frequência de CPU entre o seu valor real e o armazenado em DT (avaliar se pode ser 0)

    B = 10_000_000 # Banda do sistema
    o2 = 1*(10**-9) # Ruido aditivo Gaussiano branco de potência
    p = 0.1 # Potencia de transmissão do usuário

    B0 = 1 # ganho de potencia em referencia a distancia de um(1) metro
    distancia_user_modulo_elev_quad = (Ux - Xi)**2 + (Uy - Yi)**2 # distancia até o usuário ||distancia||  elevado ao quadrado
    if distancia_user_modulo_elev_quad == 0: distancia_user_modulo_elev_quad = 1
    h = B0/(distancia_user_modulo_elev_quad) # ganho de potencia do canal na LoS(linha de visão)
    RmjUAV = B*log2(1+((p*h)/o2)) # taxa de transmissão do user para o drone (ou do drone para o user?)
    # print(RmjUAV)
    TmjUAV_trans = Dm/RmjUAV

    # estimado_Tm = (Dm*Cm)/Fm_DT
    # computing_time_gap = (Dm*Cm*Fm_desvio)/(Fm_DT*(Fm_DT-Fm_desvio))
    TmjUAV = 0 # estimado_Tm + computing_time_gap
    Ehov = Ph*(TmjUAV_trans + TmjUAV)
    return Ehov, Ph*TmjUAV_trans, Ph*TmjUAV


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


GRID_SIZE = 70
TOTAL_STATES = 4
class AgenteRL(gym.Env):
    def __init__(self, max_clients = 25, grid_size = GRID_SIZE, coverage_radius=10):
        super(AgenteRL, self).__init__()
        
        self.coverage_radius = coverage_radius
        self.new_user_prob = 0.12
        self.user_disappearance_prob = 0.04
        self.maxclients = max_clients
        self.grid_size = grid_size
        
        self.max_consecutive_positive_rewards = 400
        self.max_consecutive_negative = 700
        # self.action_space = spaces.Discrete(5)
        self.max_battery_inital = 100
        self.battery = self.max_battery_inital
        self.action_space = gym.spaces.Discrete(41)  
        self.observation_space = spaces.Box(low=0, high=100, shape=(703,), dtype=float)
        # spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.float32)  #Imagem com 3 cores rgb(transformar em uma matriz com 0 ou 1, contendo as posições q possuem
        
        self.reset()
        
    def _get_state(self):
        # state = [
        #     x_drone, y_drone, battery,  # Posição do drone
        #     x_1, y_1, dist_1, dir_1_x, dir_1_y, estado, tempo,  # Primeiro usuário
        #     x_2, y_2, dist_2, dir_2_x, dir_2_y, estado, tempo,# Segundo usuário
        #     # ... até o máximo de usuários
        #     0, 0, 0, 0,  # Padding para slots não usados
        # ]
        num_max_clientes = 100
        num_status = 7
        state_system = np.zeros(3 + num_max_clientes * num_status)

        # Normalização da posição e bateria
        state_system[0] = self.posicao[1] / self.grid_size  # Normaliza as posições x e y do drone
        state_system[1] = self.posicao[0] / self.grid_size
        state_system[2] = self.battery / self.max_battery_inital  # Normaliza a bateria
        if len(self.users_time) == 0: max_time = 1
        else: max_time = max(self.users_time)
        if max_time == 0: max_time = 1

        index = 0
        start_idx = 3
        for i, u in enumerate(self.user_states):
            if u in [2, 3]:  # Considera os estados 2 e 3
                # Cálculo da distância e direção
                distance = np.sqrt((self.users_positions[i][0] - self.posicao[0])**2 + (self.users_positions[i][1] - self.posicao[1])**2)
                direction_y = self.users_positions[i][0] - self.posicao[0]
                direction_x = self.users_positions[i][1] - self.posicao[1]
                
                # Normalização
                norm_distance = distance / np.sqrt(2 * (self.grid_size ** 2))  # Normaliza a distância pela diagonal da grade
                norm_direction_x = direction_x / self.grid_size  # Normaliza a direção x
                norm_direction_y = direction_y / self.grid_size  # Normaliza a direção y
                norm_time = self.users_time[i] / max_time # Normaliza o tempo do usuário
                
                state_system[start_idx:start_idx + num_status] = [
                    self.users_positions[i][1] / self.grid_size,  # Normaliza a posição x do usuário
                    self.users_positions[i][0] / self.grid_size,  # Normaliza a posição y do usuário
                    norm_distance,
                    norm_direction_x,
                    norm_direction_y,
                    u / TOTAL_STATES,  # Estado do usuário (2 ou 3)
                    norm_time
                ]
                index += 1
                start_idx = 3 + index * num_status

        # state_normalized = [
        #     50/100, 50/100,      # Posição normalizada
        #     20/100,              # Distância normalizada
        #     10/100, -10/100      # Direção normalizada
        # ]
        return np.concatenate([self.clientes_grid.flatten(), state_system])
        # return state_system
        
    def get_positions(self):
        if self.index_min_previous == -1: return self.posicao, self.posicao
        return self.posicao, self.users_positions[self.index_min_previous]

    def step(self, action):
        
        self.undeterministic_random_movement()
        
        acao, penalty = self.take_action(action)

        reward, total_time, reqs, u_waiting = self.get_reward(penalty)

        # self.observation, info = self.get_observation()  #ambiente deveria mudar

       
        
        # if reward > 0:
        #     self.consecutive_positive_rewards += 1
        #     self.consecutive_negative_rewards = 0
        # elif reward < 0:
        #     self.consecutive_positive_rewards = 0
        #     self.consecutive_negative_rewards += 1
            
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
                if np.random.rand() < 0.003:
                    self.clientes_grid[self.users_positions[index][0]][self.users_positions[index][1]] = 2
                    self.user_states[index] = 2
                    self.users_time[index] = 1
                    
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
        
        
    def get_next_position(self, position, action):
        direction = action // 5  # Determina a direção (0 a 7)
        speed = (action % 5) + 1  # Velocidade de 1 a 5
        next_position = position.copy()
        if action == 40:  # Ficar parado
            speed = 0
        else:
            # Movimentos baseados na direção
            if direction == 0:  # Norte
                next_position[0] = max(0, self.posicao[0] - speed)
            elif direction == 1:  # Sul
                next_position[0] = min(self.grid_size - 1, self.posicao[0] + speed)
            elif direction == 2:  # Leste
                next_position[1] = min(self.grid_size - 1, self.posicao[1] + speed)
            elif direction == 3:  # Oeste
                next_position[1] = max(0, self.posicao[1] - speed)
            elif direction == 4:  # Nordeste
                next_position[0] = max(0, self.posicao[0] - speed)
                next_position[1] = min(self.grid_size - 1, self.posicao[1] + speed)
            elif direction == 5:  # Noroeste
                next_position[0] = max(0, self.posicao[0] - speed)
                next_position[1] = max(0, self.posicao[1] - speed)
            elif direction == 6:  # Sudeste
                next_position[0] = min(self.grid_size - 1, self.posicao[0] + speed)
                next_position[1] = min(self.grid_size - 1, self.posicao[1] + speed)
            elif direction == 7:  # Sudoeste
                next_position[0] = min(self.grid_size - 1, self.posicao[0] + speed)
                next_position[1] = max(0, self.posicao[1] - speed)
        return next_position
    
    def take_action(self, action):
        last_position = self.posicao.copy()

        speed = (action % 5) + 1
        self.posicao = self.get_next_position(last_position, action)
        # elif action == 4:
            # pass  # Ficar parado
        energy_COST = energia_voo(self.posicao, last_position, speed)
        # print(speed, energy_COST)
        # self.battery -= speed/10
        energy_penalty = energy_COST
        self.battery -= energy_COST/50
        return self.posicao, energy_penalty
    
    def get_reward(self, energy_penalty):
        reward = 0
        users_waiting = 0
        sum_time = 0
        qty = 0
        # media = np.mean(self.users_time)
        aguardando = 0
        n_pegou = 0
        energy_sobrevoo = 0
        min_distance = 100000
        index_min = -1
        for i in range(len(self.users_positions)):
            current_distance = np.linalg.norm(self.posicao - self.users_positions[i])

            if self.user_states[i] == 3:  # Usuário aguardando
                tempo_esperando = self.users_time[i]
                if current_distance <= self.coverage_radius:
                    reward += 55 - (tempo_esperando * 0.1)  # Recompensa maior por menor tempo de espera
                    self.sum_accepts += 1
                else:
                    reward -= 4.5 + (tempo_esperando * 0.05)  # Penalidade maior por rejeição com tempo de espera
                    self.sum_rejects += 1
                self.user_states[i] = 1
                self.users_time[i] = 0

            elif self.user_states[i] == 2:  # Usuário ativo
                if current_distance < min_distance:
                    index_min = i
                    min_distance = current_distance
                aguardando += 1
                if current_distance <= self.coverage_radius:
                    self.user_states[i] = 3
                    sum_time += self.users_time[i]
                    users_waiting += 1
                    qty += 1
                    # reward += 235 - (self.users_time[i] * 0.1)  # Recompensa maior por tempo de espera reduzido

                    cost_voo, _, _ = energia_sobrevoo(self.posicao, self.users_positions[i])
                    energy_sobrevoo += cost_voo
                    self.battery -= cost_voo / 20
                else:
                    n_pegou += 1
                    self.users_time[i] += 1
                    # reward -= self.users_time[i] * 0.005  # Penalidade por tempo de espera
                    # reward -= self.users_time[i] / 5000 
                    
            

            # # Comparar a distância atual com a anterior
            # distance_diff = self.previous_distances[i] - current_distance
            # if distance_diff > 0:
            #     reward += 2  # Recompensa se a distância diminuiu
            # else:
            #     reward -= 5  # Penalidade se a distância aumentou ou ficou igual

            # Atualiza a distância anterior
            if i < 100: self.previous_distances[i] = current_distance
            # else: print('I maior Q 99')
        
        if users_waiting > 0:reward += ( users_waiting / (users_waiting + n_pegou) ) * 3
        if index_min != -1 and index_min == self.index_min_previous:
            distance_diff = self.previous_distances[index_min] - min_distance
            if distance_diff > 0:
                reward += 20  # Recompensa se a distância diminuiu
            else:
                reward -= 0.3  # Penalidade se a distância aumentou ou ficou igual     
        self.index_min_previous = index_min

        if energy_penalty == 0 and aguardando == n_pegou:
            reward -= 1
        energy_penalty = energy_penalty + energy_sobrevoo
        reward = reward - (energy_penalty / 2500)
        #   else:
        #     reward = (reward*(energy_penalty))/(len(self.user_states)+1)
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
            
        # definindo usuário inicial como estado 2
        user2 = np.random.randint(0, random_clients_qty)
        self.user_states[user2] = 2
        num_max_clientes = 100
        self.previous_distances = np.zeros(num_max_clientes)
        self.index_min_previous = -1
            
        self.state = self._get_state()
        return self.state
    
    def render(self, screen, episode, total_reward, step, epsilon, confidence = 1.0, list_positions = None):
        
        screen.fill((255, 255, 255))  # Limpa a tela com branco
        block_size = 8  # Tamanho de cada bloco no grid
        color = (0,0, 255)
        for i in range(0, GRID_SIZE):
            for y in range(0,GRID_SIZE):
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
        
        # Desenhar a linha entre a posição anterior e a nova posição
        if list_positions:
            previous_position = list_positions[0]
            for i, v in enumerate(list_positions):
                if i > 0:
                    next_position = list_positions[i]
                    # if np.all(previous_position == next_position): print('IGUALLLL')
                    # print("denhsa")
                    pygame.draw.line(screen, (255, 0, 0), [previous_position[1] * block_size, previous_position[0] * block_size], [next_position[1] * block_size, next_position[0] * block_size], 10)  # Cor vermelha, largura da linha 2 pixels
                    previous_position = next_position

        # Adiciona texto de status
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Ep: {episode} | Step: {step} | Tt Rw: {total_reward:9.4f} | batt: {self.battery:9.4f} || e:{epsilon:9.4f} | cnf: {confidence}', True, (0, 0, 0))
        screen.blit(text, (10, self.grid_size * block_size + 10))

        # time.sleep(0.2)
        pygame.display.flip()  # Atualiza a tela
        

    def play_step(self):
        pass
        
    def seed(self, seed=None):
        pass

    def close(self):
        pass