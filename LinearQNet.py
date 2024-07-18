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

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = 'model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        print("FileName", file_name)
        torch.save(self.state_dict(), 'model/model.pth')


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, memory=False):
        
        if memory: 
            # print(state)
            # print(next_state)
            state = torch.stack(state, dim=0).float()
            # print(state)
            next_state = torch.stack(next_state, dim=0).float()
            # print(next_state)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
        # done = (done,)
        
        # print(range(len(done)))
        # (n, x)
        # print(reward)
        # print(action.shape)
        # print(state.shape)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.epsilon_decay = 0.95
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(10003, 256, 21)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, ambiente):
        estados = ambiente._get_state()
        # print(estados, len(estados))
        # return estados
        # estados2 = torch.tensor(estados, dtype=torch.float)
        # print(estados2.shape)
        return torch.tensor(estados, dtype=torch.float)

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        
        #MOVIMENTOS: ESQUEDA, DIREITA E FICAR PARADO
        # self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0, 0]
        if np.random.rand() < self.epsilon:
            move = random.randint(0, 20)
            final_move[move] = 1
        else:
            # state0 = torch.tensor(state, dtype=torch.float)
            # print("state0:", state0)
            prediction = self.model(state)
            # print(prediction)
            move = torch.argmax(prediction).item()
            # print(move)
            final_move[move] = 1
        if self.epsilon > 0.1: self.epsilon = self.epsilon*self.epsilon_decay
        
        return final_move


def train(plotar = False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AgenteRL()
    
    episode = 0
    tempo = 0
    score = 0
    total_reward = 0
    
    #users data
    # total_time = 0
    
    if(plotar): screen = initialize_graph(game.grid_size)
    while True:
        # get old state
        # tm.sleep(0.3)
        if plotar: game.render(screen, episode, total_reward, tempo)
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        movement = np.argmax(final_move)

        # perform move and get new state
        _, reward, done, score, info = game.step(movement)
        # total_time = info['tempos']
        score = reward
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        tempo += 1
        total_reward += reward

        if done or tempo > 1500:

            print("Done:" , done)
            # train long memory, plot result
            game.reset()
            episode += 1
            tempo = 0
            agent.n_games += 1
            agent.train_long_memory()

            if info['qtdw'] > 0 : media_tempo = info['tempos']/info['qtdw'] 
            else: media_tempo = 0

            if media_tempo > record:
                record = media_tempo
            #     agent.model.save()

            print("Media TEMPO: ", media_tempo ,info)
            print('Game', agent.n_games, 'Score', score, 'Total RW:', total_reward, 'Record:', record)
            total_reward = 0

            # score = total_time
            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            
            if(episode == 20): 
                print("SAVING MODEL")
                agent.model.save()
                break
            
    if plotar: pygame.quit()

def test():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    # agent = load
    agent = Agent()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.model.load_state_dict(torch.load('/model/model.pth', map_location=device))
    game = AgenteRL()
    
    episode = 0
    tempo = 0
    score = 0
    total_reward = 0
    
    screen = initialize_graph(game.grid_size)

    while True:
        tm.sleep(0.1)
        # get old state
        game.render(screen, episode, score, tempo)
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        movement = np.argmax(final_move)

        # perform move and get new state
        _, reward, done, score, info = game.step(movement)
        score = reward
        # state_new = agent.get_state(game)
        tempo += 1
        total_reward += reward

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            episode += 1
            game.reset()
            agent.n_games += 1
            print('Game', agent.n_games, 'Score', score, 'TOTAL RW(test):', total_reward, 'Record:', record) 
            total_reward = 0
            if episode == 100: break
    pygame.quit()

if __name__ == '__main__':
    train(plotar=True)
    test()