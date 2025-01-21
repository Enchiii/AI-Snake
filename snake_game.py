import random
from collections import namedtuple, deque
from enum import Enum

import pygame


class SnakeGame:

    Point = namedtuple("Point", "x, y")

    class Direction(Enum):
        RIGHT = 2
        LEFT = 0
        UP = 1
        DOWN = 3

    # rgb colors
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)

    SPEED = 120

    BLOCK_SIZE = 20

    def __init__(self, w, h, display, clock, font):
        self.w = w
        self.h = h

        self.display = display
        self.clock = clock
        self.font = font

        self.direction = SnakeGame.Direction.RIGHT
        self.head = SnakeGame.Point(
            (self.w // 2) // SnakeGame.BLOCK_SIZE * SnakeGame.BLOCK_SIZE,
            (self.h // 2) // SnakeGame.BLOCK_SIZE * SnakeGame.BLOCK_SIZE,
        )
        self.snake = [
            self.head,
            SnakeGame.Point(self.head.x - SnakeGame.BLOCK_SIZE, self.head.y),
            SnakeGame.Point(self.head.x - (2 * SnakeGame.BLOCK_SIZE), self.head.y),
        ]
        self.frame_iteration = 0
        self.score = 0
        self.food = None
        self._place_food()

    def reset(self):
        self.direction = SnakeGame.Direction.RIGHT
        self.head = SnakeGame.Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            SnakeGame.Point(self.head.x - SnakeGame.BLOCK_SIZE, self.head.y),
            SnakeGame.Point(self.head.x - (2 * SnakeGame.BLOCK_SIZE), self.head.y),
        ]
        self.frame_iteration = 0
        self.score = 0
        self.food = None
        self._place_food()
        # self.move_history_len = self.w // SnakeGame.BLOCK_SIZE if self.w > self.h else self.h // SnakeGame.BLOCK_SIZE
        # self.move_history = deque(maxlen=self.move_history_len)

    def _place_food(self):
        x = random.randint(0, (self.w - SnakeGame.BLOCK_SIZE) // SnakeGame.BLOCK_SIZE) * SnakeGame.BLOCK_SIZE
        y = random.randint(0, (self.h - SnakeGame.BLOCK_SIZE) // SnakeGame.BLOCK_SIZE) * SnakeGame.BLOCK_SIZE
        self.food = SnakeGame.Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False

        if self.frame_iteration > 100 * len(self.snake):
            reward = -8

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        clock_wise = [
            SnakeGame.Direction.RIGHT,
            SnakeGame.Direction.DOWN,
            SnakeGame.Direction.LEFT,
            SnakeGame.Direction.UP,
        ]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            self.direction = clock_wise[idx]
        elif action == [0, 1, 0]:
            self.direction = clock_wise[(idx + 1) % 4]
        elif action == [0, 0, 1]:
            self.direction = clock_wise[(idx - 1) % 4]

        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        if self._is_collision():
            game_over = True
            reward = -16
            return game_over, self.score, reward

        # 4. place new food or just move
        if self.head == self.food:
            self.frame_iteration = 0
            self.score += 1
            reward = +16
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SnakeGame.SPEED)
        # 6. return game over and score
        return game_over, self.score, reward

    def _is_collision(self, pt=None):
        if not pt:
            pt = self.head
        # hits boundary
        if pt.x > self.w - SnakeGame.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - SnakeGame.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(SnakeGame.BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, SnakeGame.BLUE1, pygame.Rect(pt.x, pt.y, SnakeGame.BLOCK_SIZE, SnakeGame.BLOCK_SIZE)
            )
            pygame.draw.rect(self.display, SnakeGame.BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(
            self.display,
            SnakeGame.RED,
            pygame.Rect(self.food.x, self.food.y, SnakeGame.BLOCK_SIZE, SnakeGame.BLOCK_SIZE),
        )

        text = self.font.render("Score: " + str(self.score), True, SnakeGame.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == SnakeGame.Direction.RIGHT:
            x += SnakeGame.BLOCK_SIZE
        elif direction == SnakeGame.Direction.LEFT:
            x -= SnakeGame.BLOCK_SIZE
        elif direction == SnakeGame.Direction.DOWN:
            y += SnakeGame.BLOCK_SIZE
        elif direction == SnakeGame.Direction.UP:
            y -= SnakeGame.BLOCK_SIZE

        self.head = SnakeGame.Point(x, y)

    def get_state(self):
        head = self.head
        point_l = SnakeGame.Point(head.x - 20, head.y)
        point_r = SnakeGame.Point(head.x + 20, head.y)
        point_u = SnakeGame.Point(head.x, head.y - 20)
        point_d = SnakeGame.Point(head.x, head.y + 20)

        dir_l = self.direction == SnakeGame.Direction.LEFT
        dir_r = self.direction == SnakeGame.Direction.RIGHT
        dir_u = self.direction == SnakeGame.Direction.UP
        dir_d = self.direction == SnakeGame.Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # Danger right
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # Danger left
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # food in front
            (dir_d and self.food.y > self.head.y)
            or (dir_u and self.food.y < self.head.y)
            or (dir_r and self.food.x > self.head.x)
            or (dir_l and self.food.x < self.head.x),
            # food on right
            (dir_d and self.food.x > self.head.x)
            or (dir_u and self.food.x < self.head.x)
            or (dir_r and self.food.y > self.head.y)
            or (dir_l and self.food.y < self.head.y),
            # food on left
            (dir_d and self.food.x < self.head.x)
            or (dir_u and self.food.x > self.head.x)
            or (dir_r and self.food.y < self.head.y)
            or (dir_l and self.food.y > self.head.y),
            # food behind
            (dir_d and self.food.y < self.head.y)
            or (dir_u and self.food.y > self.head.y)
            or (dir_r and self.food.x < self.head.x)
            or (dir_l and self.food.x > self.head.x),
            # snake direction
            dir_d,
            dir_u,
            dir_r,
            dir_l,
        ]

        return state
