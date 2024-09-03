import torch
import random
import numpy as np
from collections import deque
from AgenteRL import AgenteRL
from plot_helper import plot, initialize_graph

import pygame
import time as tm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys

QTD_MOVEMENT = 41
versao = 40 # 500
LAST_MODEL = 'remember_dist_-CNNLsTM5600_41_normalized_model' + str(versao) + '.pth'
PREVISION_LENGTH = 3
class Linear_QNet(nn.Module):
    def __init__(self, rnn_hidden_size, cnn_output_size, system_state_size,  intermediate_linear, output_size, grid_size=70, num_layers=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        
        self.flatten_size = 128 * 17 * 17
        self.fc1 = nn.Linear(self.flatten_size, cnn_output_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=cnn_output_size + system_state_size, hidden_size=rnn_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_instermediate = nn.Linear(rnn_hidden_size, intermediate_linear)
        self.fc_out = nn.Linear(intermediate_linear, output_size)
        

    def forward(self, grid, system, hidden=None):
        
        if grid.dim() == 2:
            grid = grid.unsqueeze(0).unsqueeze(0)  # De [70, 70] para [1, 1, 70, 70]
        elif grid.dim() == 3:
            grid = grid.unsqueeze(1)  
        
        # Passar o grid pela CNN
        x = F.relu(self.conv1(grid))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        
        if system.dim() == 1:
            system = system.unsqueeze(0)  # De [703] para [1, 703]

        x_combined = torch.cat((x, system), dim=1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x_combined.unsqueeze(1), hidden)
        # print(lstm_out)
        out = self.fc_instermediate(lstm_out[:, -1, :])
        out = self.fc_out(out)

        return out, hidden

    def save(self, file_name='model.pth'):
        model_folder_path = 'model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        path_name = file_name
        file_name = os.path.join(model_folder_path, file_name)
        print("FileName", file_name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_save = os.path.join(current_dir, 'model', 'remember_dist_-CNNLsTM5600_' + str(QTD_MOVEMENT) + '_normalized_'+ str(path_name))
        torch.save(self.state_dict(), path_to_save)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, memory=False):

        # clientes_grid = torch.tensor(clientes_grid, dtype=torch.float32)
        # state_system = torch.tensor(state_system, dtype=torch.float32)
        # next_clientes_grid = torch.tensor(next_clientes_grid, dtype=torch.float32)
        # next_state_system = torch.tensor(next_state_system, dtype=torch.float32)
        # # print(clientes_grid.shape, state_system.shape, next_clientes_grid.shape)
        if memory:
            # print(state.shape)
            
            # Para treinamento com memória, os estados já devem estar empilhados corretamente
            clientes_grid, state_system = zip(*state)
            next_clientes_grid, next_state_system = zip(*next_state)
            
            # clientes_grid = torch.tensor(clientes_grid, dtype=torch.float32)
            # state_system = torch.tensor(state_system, dtype=torch.float32)
            # next_clientes_grid = torch.tensor(next_clientes_grid, dtype=torch.float32)
            # next_state_system = torch.tensor(next_state_system, dtype=torch.float32)

            # Converter listas de tensores em tensores únicos
            clientes_grid = torch.stack(clientes_grid, dim=0).float()
            state_system = torch.stack(state_system, dim=0).float()
            next_clientes_grid = torch.stack(next_clientes_grid, dim=0).float()
            next_state_system = torch.stack(next_state_system, dim=0).float()
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
        else:
            # Separar o estado em grid de clientes e estado do sistema
            clientes_grid, state_system = state
            # print(clientes_grid)
            next_clientes_grid, next_state_system = next_state
            
            # clientes_grid = torch.tensor(clientes_grid, dtype=torch.float32)
            # state_system = torch.tensor(state_system, dtype=torch.float32)
            # next_clientes_grid = torch.tensor(next_clientes_grid, dtype=torch.float32)
            # next_state_system = torch.tensor(next_state_system, dtype=torch.float32)
            clientes_grid = clientes_grid.clone().detach().float()
            state_system = state_system.clone().detach().float()
            next_clientes_grid = next_clientes_grid.clone().detach().float()
            next_state_system = next_state_system.clone().detach().float()
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

        # Adicionar dimensão de sequência se necessário
        if len(clientes_grid.shape) == 2:  # Assumindo (batch_size, grid_size1, grid_size2)
            clientes_grid = clientes_grid.unsqueeze(0)
            next_clientes_grid = next_clientes_grid.unsqueeze(0)
            state_system = state_system.unsqueeze(0)
            next_state_system = next_state_system.unsqueeze(0)
            reward = reward.unsqueeze(0)

        done = (done,)

        # Inicializar o estado oculto da LSTM
        hidden = None

        # Prever os valores Q com o estado atual
        pred, hidden = self.model(clientes_grid, state_system, hidden)

        # Copiar o tensor de predição para modificá-lo com os valores-alvo
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_pred, _ = self.model(next_clientes_grid[idx].unsqueeze(0), next_state_system[idx].unsqueeze(0), hidden)
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Zerar os gradientes, realizar o backward e atualizar os pesos
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

MAX_MEMORY = 100_000
BATCH_SIZE = 4000
LR_anterior = 0.002
LR = 0.0007

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9 # randomness
        self.epsilon_decay = 0.9985
        self.gamma = 0.85 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(300, 300, 703, 120, QTD_MOVEMENT)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, ambiente):
        grid, system = ambiente._get_state()
        # print(estados, len(estados))
        # return estados
        # estados2 = torch.tensor(estados, dtype=torch.float)
        # print(estados2.shape)
        grid = torch.tensor(grid, dtype=torch.float32)
        system = torch.tensor(system, dtype=torch.float32)
        return grid, system

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, memory=True)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, test=False):
        # random moves: tradeoff exploration / exploitation
        
        #MOVIMENTOS: ESQUEDA, DIREITA E FICAR PARADO
        # self.epsilon = 80 - self.n_games
        # state = (state - np.mean(state)) / np.std(state)
        final_move = [0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0, 0] # ADD=> [,   0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0, 0]
        if np.random.rand() < self.epsilon:
            move = random.randint(0, QTD_MOVEMENT-1)
            final_move[move] = 1
            confidence = -1
        else:
            # state0 = torch.tensor(state, dtype=torch.float)
            # print("state0:", state0)
            grid, system = state
            # grid = torch.tensor(grid, dtype=torch.float32)
            # system = torch.tensor(system, dtype=torch.float32)1
            # teste = torch.randn(1, 70, 70)
            # print(grid.shape, teste.shape)
            prediction, _ = self.model(grid, system)
            # print('predicaooo:::>>', prediction)
            move = torch.argmax(prediction).item()
            # print('movimento:', move)
            # if (test):print("move:", move)
            
            # print('prediction::', prediction)
            
            probabilities = torch.softmax(prediction[0], dim=0)  # Convertendo logits para probabilidades
            
            confidence = probabilities[int(move)].item() 
            # print(confidence)
            #sys.stdout.write(str(probabilities))
            #sys.stdout.write(str(confidence))
            #sys.stdout.flush()
            #
            # print(probabilities, confidence)
            # print(move)
            final_move[int(move)] = 1
        # if self.epsilon > 0.1 and test==False: self.epsilon = self.epsilon*self.epsilon_decay
        # print("final move:", final_move)
        return final_move, confidence


def definir_action(drone_position, user_position):
    # Calcular distâncias
    dist_x = user_position[0] - drone_position[0]
    dist_y = user_position[1] - drone_position[1]

    # Determinar direção
    if dist_x < 0 and dist_y == 0:
        direction = 0  # Norte
    elif dist_x > 0 and dist_y == 0:
        direction = 1  # Sul
    elif dist_x == 0 and dist_y > 0:
        direction = 2  # Leste
    elif dist_x == 0 and dist_y < 0:
        direction = 3  # Oeste
    elif dist_x < 0 and dist_y > 0:
        direction = 4  # Nordeste
    elif dist_x < 0 and dist_y < 0:
        direction = 5  # Noroeste
    elif dist_x > 0 and dist_y > 0:
        direction = 6  # Sudeste
    elif dist_x > 0 and dist_y < 0:
        direction = 7  # Sudoeste

    # Calcular velocidade
    speed = min(max(abs(dist_x), abs(dist_y)), 5)

    # Definir a ação
    if np.all(user_position == drone_position):
        action = 40  # Ficar parado
    else:
        action = direction * 5 + (speed - 1)

    return action

def train(plotar = False, continuar = False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AgenteRL()
    
    if continuar == True:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model', LAST_MODEL)
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
    
    episode = 0
    tempo = 0
    score = 0
    total_reward = 0
    decay_epsilon = 0
    
    agent.epsilon = 0.9
    
    #users data
    # total_time = 0
    confiancaaa = -1
    
    if(plotar): screen = initialize_graph(game.grid_size)
    while True:
        # get old state
        epsilon = agent.epsilon
        if plotar: game.render(screen, episode, total_reward, tempo, epsilon, confidence=confiancaaa)
        state_old = agent.get_state(game)

        # get move
        final_move, confiancaaa = agent.get_action(state_old, test=False)
        movement = np.argmax(final_move)

        # perform move and get new state
        _, reward, done, score, info = game.step(movement)
        # total_time = info['tempos']
        score = reward
        state_new = agent.get_state(game)

        # //// ======== ========== TRAINING short MEMORY =========== ======== \\\\\
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        # agent.remember(state_old, final_move, reward, state_new, done)
        
        
        tempo += 1
        total_reward += reward

        if done or tempo > 1300:

            print("Done:" , done)
            if episode < 80 : decay_epsilon += 1
            # train long memory, plot result
            game.reset()
            episode += 1
            tempo = 0
            agent.n_games += 1
            
            # agent.train_long_memory()

            if info['qtdw'] > 0 : media_tempo = info['tempos']/info['qtdw'] 
            else: media_tempo = 0

            if media_tempo > record:
                record = media_tempo
            #     agent.model.save()

            print("Media TEMPO: ", media_tempo ,info)
            print('Game', agent.n_games, 'Score', score, 'Total RW:', total_reward, 'Record:', record)
            total_reward = 0
            
            agent.epsilon = 0.9-(decay_epsilon / 100)
            print(decay_epsilon)
            # score = total_time
            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            
            if(episode % 20 == 0): 
                print("SAVING MODEL")
                
                # decay_epsilon = 0
                versao_modelo = episode + versao
                agent.model.save(file_name='model'+ str(versao_modelo) + '.pth')
            if episode == 1600: break
            
    if plotar: pygame.quit()
    
def simulate_next_positions(agent ,game, action, next_position):
    
    game_copy = AgenteRL()
    return_positions = []
    # game_copy
    game_copy.battery = game.battery
    # self.consecutive_positive_rewards = 0 
    # self.consecutive_negative_rewards = 0

    # self.sum_accepts = 0
    # self.sum_rejects = 0
    # game_copy.total_time_waiting = game.total_time_waiting
    game_copy.num_waiting = game.num_waiting
    
    game_copy.posicao = game.posicao.copy()
    game_copy.clientes_grid = game.clientes_grid.copy()
    game_copy.users_positions = game.users_positions.copy()
    game_copy.users_time = game.users_time.copy()
    # self.user_states = [ ]
    
    game_copy.user_states = game.user_states.copy()
    # game_copy.
    for i in range(PREVISION_LENGTH):
        game_copy.undeterministic_random_movement()
            
        acao, penalty = game_copy.take_action(action)
        
        # for i in range(0,)
        prev_pos = game_copy.posicao.copy()
        state_old = agent.get_state(game_copy)
        final_move, confidence = agent.get_action(state_old, test=True)
        movement = np.argmax(final_move)

        next_position = game_copy.get_next_position(prev_pos, movement)
        # list_positions[1] = next_position
        # print(movement)
        # if np.all(position == next_position): print('ENGUALL')
        return_positions.append(next_position)
        # print(return_positions)
    return return_positions

def test():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', LAST_MODEL)
    record = 0
    # agent = load
    agent = Agent()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    game = AgenteRL()
    
    episode = 0
    tempo = 0
    score = 0
    total_reward = 0
    
    initial_position_copy = game.posicao.copy()
    list_positions = [initial_position_copy]
    for i in range(PREVISION_LENGTH): list_positions.append(initial_position_copy)
    confidence = 0.0
    agent.epsilon = 0.05
    epsilon = agent.epsilon
    
    screen = initialize_graph(game.grid_size)

    while True:
        tm.sleep(0.1)
        tempo += 1
        # get old state
        state_old = agent.get_state(game)
        # print(state_old)
        list_positions[0] = game.posicao.copy()
        # get move
        final_move, confidence = agent.get_action(state_old, test=True)
        # print(final_move)
        movement = np.argmax(final_move)
        # print(movement)
        
        if confidence < 0.3:
            drone_pos, user_pos = game.get_positions()
            movement = definir_action(drone_pos, user_pos)
        else:
            # print(movement)
            pass
            
            
            
        # list_positions[0] = game.posicao
        # list_positions.append(game.posicao)
        next_position = game.get_next_position(game.posicao, movement)
        # list_positions[1] = next_position
        
        new_list_positions = simulate_next_positions(agent, game, movement, next_position)
        for i, v in enumerate(new_list_positions):
            # print('index', i, v)
            list_positions[i+1] = v

        game.render(screen, episode, total_reward, tempo, epsilon=epsilon, confidence = confidence, list_positions=list_positions)
        # perform move and get new state
        _, reward, done, score, info = game.step(movement)
        score = reward
        # state_new = agent.get_state(game)
        # tempo += 1
        # print(reward)
        total_reward += reward
        # if reward > 100: print(reward, " - ", total_reward)
        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            episode += 1
            tempo = 0
            game.reset()
            agent.n_games += 1
            print('Game', agent.n_games, 'Score', score, 'TOTAL RW(test):', total_reward, 'Record:', record) 
            total_reward = 0
            if episode == 100: break
    pygame.quit()

if __name__ == '__main__':
    if sys.argv[1] == 't':
        train(plotar=False)
    if sys.argv[2] == 'c':
        train(continuar=True)
    # train(continuar = True)
    test()