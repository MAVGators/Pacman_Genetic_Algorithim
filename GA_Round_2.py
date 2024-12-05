import pygame as pyg
import pacman
from pacman import Game_Setup, Game, Player, Pacman, Ghost, Maze, Pacman_Game, Genetic_Game
import time
import random
import matplotlib.pyplot as plt
import numpy as np


pyg.init()

#contain all the functions related to dealing with the genetic algorithm
class Genetics():

    def __init__(self):
        #basic setup needed to run the visuals for the game
        self.clock = pyg.time.Clock()
        self.SETUP = Game_Setup()
        self.test_gene = 'LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUULLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD'
        self.population_size = 300
        self.gene_length = 3000
        self.mutation_rate = 0.01
        self.generations = 400


    def generate_initial_population(self):
        self.population = []
        for _ in range(self.population_size):
            gene = ''.join(random.choice('UDLR') for _ in range(self.gene_length))
            self.population.append(gene)

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.gene_length - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, gene):
        gene_list = list(gene)
        for i in range(len(gene_list)):
            if random.random() < self.mutation_rate:
                gene_list[i] = random.choice('UDLR')
        return ''.join(gene_list)

    def run_genetic_algorithm(self):
        self.generate_initial_population()
        best_gene = None
        best_fitness = float('-inf')
        fitness_history = []

        # Initialize the plot
        plt.ion()
        fig, ax = plt.subplots()
        best_fitnesses = []
        avg_fitnesses = []
        worst_fitnesses = []
        line1, = ax.plot(best_fitnesses, label='Best Fitness')
        line2, = ax.plot(avg_fitnesses, label='Average Fitness')
        line3, = ax.plot(worst_fitnesses, label='Worst Fitness')
        ax.legend()

        #write the results of this simulation to a seperate file
        current_time = time.strftime("%Y%m%d-%H%M%S")
        filename = f'best_gene_{current_time}.txt'
        with open(filename, 'w') as file:
            file.write('Best genes by generation:\n')


        for generation in range(self.generations):
            start_time = time.time()
            fitness_scores = []
            for gene in self.population:

                game = self.run_game(gene, is_displaying=False)
                fitness = self.fitness(game)
                fitness_scores.append(fitness)

            best_fitness = max(fitness_scores)

            fitness_history.append(best_fitness)

            best_fitnesses.append(max(fitness_scores))
            avg_fitnesses.append(np.mean(fitness_scores))
            worst_fitnesses.append(min(fitness_scores))

            # Update the plot
            line1.set_ydata(best_fitnesses)
            line1.set_xdata(range(len(best_fitnesses)))
            line2.set_ydata(avg_fitnesses)
            line2.set_xdata(range(len(avg_fitnesses)))
            line3.set_ydata(worst_fitnesses)
            line3.set_xdata(range(len(worst_fitnesses)))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

            best_gene = self.population[fitness_scores.index(best_fitness)]
            with open(filename, 'a') as file:
                file.write(f'Generation {generation + 1}, Fitness {best_fitness}: {best_gene}\n')
            
            print(f'Generation {generation + 1} best fitness: {best_fitness}, Sim Time: {time.time() - start_time}')

            # Elitism: carry over the best genes to the new population
            num_elites = 1
            elites = [self.population[i] for i in np.argsort(fitness_scores)[-num_elites:]]

            new_population = []
            new_population.extend(elites)
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        
        plt.ioff()
        plt.show()

        return best_gene

    #run a game with a gene and return the end game object after game is over
    def run_game(self, gene, is_displaying):
        game = Genetic_Game(self.SETUP, self.clock, gene, is_displaying)
        # Main game loop
        while game.game_over_bool != True and game.ticks['game'] < self.gene_length:
            # Update the game
            game.update()
            #update the display
            if is_displaying:
                pyg.display.flip()
                self.clock.tick(100)
        return game

    #fitness function which takes in a pacman game and uses the end state of the game to determine how well the pacman did
    def fitness(self, game):
        #constants for tuning weight of parameters
        k_score = 0
        #turning off survival time for now, only care about pellets
        k_time = 0

        k_pellets = 1
        #wins are highly desired especially in the beginning where all I really want is a pacman that wins
        k_win = 100000000000

        #collect info from the game
        #final pacman score
        score = game.player.score
        #time alive
        time = game.ticks['game']
        #pellets eaten
        pellets = game.maze.pellets_eaten()
        #win or loss
        win = 1 if game.won else 0

        return k_score * score + k_time * time + k_pellets * pellets + k_win * win
    


    '''
    
    DEBUGGING AND TEST FUNCTIONS FOR THE GA
    
    '''
    def genetic_test(self, gene):
        # Create the game
        clock = pyg.time.Clock()
        best_score = 0
        is_displaying = False
        for i in range(100):
            game = Genetic_Game(self.SETUP, clock, gene, is_displaying)
            # Main game loop
            while game.game_over_bool != True:

                # Update the game
                game.update()

                #update the display
                if is_displaying:
                    pyg.display.flip()
                    clock.tick(60)

            if game.player.score > best_score:
                best_score = game.player.score

        return game.player.score



#function to get the best gene from a certain generation
def get_gene_by_generation(generation_number, filename='best_gene.txt'):
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith(f'Generation {generation_number},'):
                return line.split(': ')[1].strip()
    return None



#unit tests
def test_1():
    pg = Pacman_Game()
    gen = Genetics()
    pg.genetic_test(gen.test_gene)

def test_2():
    gen = Genetics()
    gen.genetic_test(gen.test_gene)

def test_3():
    gen_alg = Genetics()
    print(gen_alg.fitness(gen_alg.run_game(gen_alg.test_gene, is_displaying=False)))

def train():
    gen_alg = Genetics()
    print(gen_alg.run_genetic_algorithm())

def test_gene(gen, filename='best_gene.txt'):
    gen_alg = Genetics()
    gene = get_gene_by_generation(gen, filename)
    print(gen_alg.fitness(gen_alg.run_game(gene, is_displaying=True)))


def main():
    start_time = time.time()
    
    #test_1()
    #test_2()
    #test_3()
    train()
    #test_gene(168,'best_gene_20241204-200900.txt')
    end_time = time.time()


    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
