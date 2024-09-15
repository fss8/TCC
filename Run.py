import torch
import random
import numpy as np

import pygame
import time as tm

import os
import sys

from AgenteRL import AgenteRL
from LinearQNet import Agent
from Kmean import Kmean
from plot_helper import plot, initialize_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

versao = 620 # 500
QTD_MOVEMENT = 41
LEFT_NAME = 'remember_dist_-CNNLsTM5600_41_normalized_model'
LAST_MODEL = str(LEFT_NAME) + str(versao) + '.pth'

PREVISION_LENGTH = 3

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
    agent = Agent(QTD_MOVEMENT, LEFT_NAME)
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
        agent.epsilon *= agent.epsilon_decay
        
        
        tempo += 1
        total_reward += reward

        if done or tempo > 1200:

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
            
            total_score += total_reward
            agent.epsilon = 0.9-(decay_epsilon / 100)
            print_scores(decay_epsilon, total_reward, total_score, plot_scores, plot_mean_scores, agent.n_games)
            total_reward = 0
            if(episode % 20 == 0): 
                print("SAVING MODEL")
                
                # decay_epsilon = 0
                versao_modelo = episode + versao
                agent.model.save(file_name=str(versao_modelo) + '.pth')
            if episode == 1600: break
            
    if plotar: pygame.quit()
    
def print_scores(decay_epsilon, score, sum_of_rewards, plot_scores, plot_mean_scores, n_games):
    print(decay_epsilon)
    plot_scores.append(score)
    mean_score = sum_of_rewards / n_games
    plot_mean_scores.append(mean_score)
    plot(plot_scores, plot_mean_scores)
            

    
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
    agent = Agent(QTD_MOVEMENT, LEFT_NAME)
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
        print(movement)
        
        # if confidence < 0.3:
        #     drone_pos, user_pos = game.get_positions()
        #     movement = definir_action(drone_pos, user_pos)
        # else:
        #     # print(movement)
        #     pass
            
            
            
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
    
def test_with_kmean():
    kmean = Kmean()
    
    game = AgenteRL()
    game.reset()
    
    episode = 0
    tempo = 0
    score = 0
    total_reward = 0
    record = 0
    
    screen = initialize_graph(game.grid_size)
    while True:
        tm.sleep(0.2)
        drone_pos, user_pos, user_states = game.get_positions()
        # print(drone_pos, user_pos, user_states)
        # contains_two = np.any(user_states == 2)
        # if contains_two:
        #     n_clusters = max()
        #     movement = kmean.determine_next_action(drone_pos, user_pos, user_states)
        # else: movement = 40
        _, movement = kmean.determine_next_action(drone_pos, user_pos, user_states)
        # print(movement)
        
        game.render(screen, episode, total_reward, tempo, epsilon=-2, confidence = 2)
        _, reward, done, score, info = game.step(movement)
        
        score = reward
        
        total_reward += reward
        # if reward > 100: print(reward, " - ", total_reward)

        if done:
            # train long memory, plot result
            episode += 1
            tempo = 0
            game.reset()
            print('Game', episode, 'Score', score, 'TOTAL RW(test):', total_reward, 'Record:', record) 
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
    # test_with_kmean()