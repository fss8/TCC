import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

    return action, direction, speed

class Kmean:
    def __init__(self, grid_size=70):
        self.grid_size = grid_size
        # self.action_space = self.create_action_space()

    # Criando as 41 opções de movimentos
    # def create_action_space(self):
    #     movements = []
    #     for direction in range(8):  # 8 direções (N, S, L, O, NE, NO, SE, SO)
    #         for speed in range(1, 6):  # Velocidades de 1 a 5
    #             if direction == 0:  # Norte
    #                 movements.append((-speed, 0))
    #             elif direction == 1:  # Sul
    #                 movements.append((speed, 0))
    #             elif direction == 2:  # Leste
    #                 movements.append((0, speed))
    #             elif direction == 3:  # Oeste
    #                 movements.append((0, -speed))
    #             elif direction == 4:  # Nordeste
    #                 movements.append((-speed, speed))
    #             elif direction == 5:  # Noroeste
    #                 movements.append((-speed, -speed))
    #             elif direction == 6:  # Sudeste
    #                 movements.append((speed, speed))
    #             elif direction == 7:  # Sudoeste
    #                 movements.append((speed, -speed))
    #     movements.append((0, 0))  # Adiciona a ação de ficar parado
    #     return np.array(movements)

    # Calcular a direção do movimento
    def determine_next_action(self, drone_position, users_positions, user_states, n_clusters=3):
        relevant_users = [pos for i, pos in enumerate(users_positions) if user_states[i] in [2,3]]
        
        if len(relevant_users) == 0:
            return None, 40

        relevant_users = np.array(relevant_users)
        # n_clusters = min(int((3*(len(user_states)))/10), len(relevant_users))
        n_clusters = self.density_based_clusters(users_positions)
        n_clusters = min(n_clusters, len(relevant_users))

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(relevant_users)

        centroids = kmeans.cluster_centers_
        closest_centroid = min(centroids, key=lambda c: np.linalg.norm(drone_position - c))

        direction = closest_centroid - drone_position
        # print(direction)
        # norm = np.linalg.norm(direction)

        # if norm > 0:
        #     direction = direction / norm
        
        acao, next_direction, drone_speed = definir_action(drone_position, closest_centroid)

        # drone_speed = 1  # Definir a velocidade do drone
        next_position = drone_position + direction * drone_speed

        next_position[0] = np.clip(next_position[0], 0, self.grid_size - 1)
        next_position[1] = np.clip(next_position[1], 0, self.grid_size - 1)

        return next_position, acao
    
    def density_based_clusters(self, users_positions, density_threshold=5):
        """
        Define o número de clusters com base na densidade de usuários.
        Se houver mais de density_threshold usuários numa área, aumenta o número de clusters.
        """
        n_users = len(users_positions)
        
        # Calcular densidade de usuários por área
        density = n_users / (self.grid_size ** 2)
        
        # Ajustar número de clusters baseado na densidade
        if density > density_threshold:
            return min(10, n_users)  # Ajusta o número máximo de clusters para não exceder o número de usuários
        else:
            return max(2, n_users // density_threshold)  # Ajusta o número mínimo para evitar poucos clusters

    # # Encontrar a ação mais próxima nas 41 opções
    # def find_nearest_action(self, dist_x, dist_y):
    #     movement_vector = np.array([dist_x, dist_y])
    #     distances = np.linalg.norm(self.action_space - movement_vector, axis=1)
    #     nearest_action_idx = np.argmin(distances)
    #     return nearest_action_idx

# Plotar as posições do drone e dos usuários
def plot_positions(drone_position, next_position, users_positions):
    plt.figure(figsize=(7, 7))
    plt.xlim(0, 70)
    plt.ylim(0, 70)

    # Plotando a posição atual e a próxima posição do drone
    print(drone_position)
    next_position = np.array(next_position, dtype=int)
    print(next_position)
    plt.scatter(*drone_position, color='blue', label='Posição Inicial do Drone')
    

    # Plotando as posições dos usuários
    for i, user_pos in enumerate(users_positions):
        plt.scatter(*user_pos, color='red', label=f'Usuário {i+1}' if i == 0 else "")
    
    plt.scatter(*next_position, color='black', label='Próxima Posição do Drone')

    plt.legend()
    plt.grid(True)
    plt.title('Movimento do Drone e Posições dos Usuários')
    plt.show()