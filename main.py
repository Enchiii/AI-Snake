import random
import sys

import pygame
import numpy as np

from agent import Agent
from snake_game import SnakeGame
from helper import plot

pygame.init()

WIDTH = 640  # Must be divisible by 40
HEIGHT = 480  # Must be divisible by 40
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 25)


if __name__ == "__main__":
    agent = Agent(epsilon=0.8)
    game = SnakeGame(WIDTH, HEIGHT, display, clock, font)
    scores = []
    mean_scores = []
    target_update = 30

    # game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.K_q:
                pygame.quit()
                sys.exit()

        state = game.get_state()

        action = agent.get_action(state)

        game_over, score, reward = game.play_step(list(action))

        new_state = game.get_state()

        agent.update_memory(state, action, reward, new_state, game_over)

        agent.train()

        if game_over:
            agent.n_games += 1
            scores.append(score)
            mean_scores.append(np.mean(scores))
            if score > agent.record:
                agent.record = score
                agent.model.save()
            game.reset()

            print(f"Game {agent.n_games}, Score: {score}, Record {agent.record}, Mean {np.mean(mean_scores)}")

            plot(scores, mean_scores)

        if agent.n_games % target_update == 0:
            agent.lr *= random.uniform(0.9, 1.1)
            agent.update_target_model()
