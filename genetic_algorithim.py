import pygame as pyg
import pacman
from pacman import Game, Player, Pacman, Ghost, Maze, Pacman_Game
import time


def main():
    start_time = time.time()
    pg = Pacman_Game()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
