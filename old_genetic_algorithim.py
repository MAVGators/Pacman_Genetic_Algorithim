import pygame as pyg
import pacman
from pacman import Game_Setup, Game, Player, Pacman, Ghost, Maze, Pacman_Game, Genetic_Game
import time
import random
import matplotlib.pyplot as plt


pyg.init()

#contain all the functions related to dealing with the genetic algorithm
class Genetics():

    def __init__(self):
        #basic setup needed to run the visuals for the game
        self.clock = pyg.time.Clock()
        self.SETUP = Game_Setup()
        self.test_gene = 'LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUULLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD'
        self.population_size = 10
        self.gene_length = 6000
        self.mutation_rate = 0.01
        self.generations = 200


    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            gene = ''.join(random.choice('LRUD') for _ in range(self.gene_length))
            population.append(gene)
        return population
    
    def select_parents(self, population, fitness_scores):
        sorted_population = [gene for _, gene in sorted(zip(fitness_scores, population), reverse=True)]
        return sorted_population[:2]
    
    def mutate(self, gene):
        gene_list = list(gene)
        for i in range(len(gene_list)):
            if random.random() < self.mutation_rate:
                gene_list[i] = random.choice('LRUD')
        return ''.join(gene_list)
    
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.gene_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    

    def run_genetic_algorithm(self):
        population = self.initialize_population()
        best_fitness_scores = []
        average_fitness_scores = []

        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(best_fitness_scores, label='Best Fitness')
        line2, = ax.plot(average_fitness_scores, label='Average Fitness')
        ax.legend()

        current_time = time.strftime("%Y%m%d-%H%M%S")
        filename = f'best_gene_{current_time}.txt'
        with open(filename, 'w') as file:
            file.write('Best genes by generation:\n')
        
        for generation in range(self.generations):
            fitness_scores = [self.fitness(self.run_game(gene, is_displaying=False)) for gene in population]
            best_fitness = max(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_scores.append(best_fitness)
            average_fitness_scores.append(average_fitness)

            best_gene = population[fitness_scores.index(best_fitness)]
            with open(filename, 'a') as file:
                file.write(f'Generation {generation + 1}, Fitness {best_fitness}: {best_gene}\n')

            new_population = []
            # Elitism: carry the best gene to the next generation
            new_population.append(best_gene)
            for _ in range((self.population_size - 1) // 2):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            population = new_population

            print(f'Generation {generation + 1} best fitness: {best_fitness}')

            line1.set_ydata(best_fitness_scores)
            line1.set_xdata(range(len(best_fitness_scores)))
            line2.set_ydata(average_fitness_scores)
            line2.set_xdata(range(len(average_fitness_scores)))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        plt.ioff()
        plt.show()

        return best_gene




    #run a game with a gene and return the end game object after game is over
    def run_game(self, gene, is_displaying):
        game = Genetic_Game(self.SETUP, self.clock, gene, is_displaying)
        # Main game loop
        while game.game_over_bool != True:
            # Update the game
            game.update()
            #update the display
            if is_displaying:
                pyg.display.flip()
                self.clock.tick(60)
        return game

    #fitness function which takes in a pacman game and uses the end state of the game to determine how well the pacman did
    def fitness(self, game):
        #constants for tuning weight of parameters
        k_score = 0
        k_time = 0.1
        k_pellets = 10
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

def test_4():
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
    test_4()
    #test_gene(28,'best_gene_20241204-082811.txt')
    
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
