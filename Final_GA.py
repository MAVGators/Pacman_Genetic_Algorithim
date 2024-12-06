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
        self.test_gene = 'LLUULLDD'

        #genetic algorithm parameters
        self.population_size = 200
        self.gene_length = 1000
        self.starting_mutation_rate = 0.08
        self.muation_rate_decay = 0
        self.mutation_rate = self.starting_mutation_rate
        self.generations = 1000

    #generate the initial population of random genes
    def generate_initial_population(self):
        self.population = []
        for _ in range(self.population_size):
            gene = ''.join(random.choice('UDLR') for _ in range(self.gene_length))
            self.population.append(gene)

    #implement a forced initial population to test the GA, this is a simple gene that already gets a good score (developed by a previous GA)
    #and the training on this will refine the solution
    def forced_initial_population(self, gene):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(gene)

    #select two parents from the population based on their fitness scores
    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents

    #'breed' the two parents to create a child gene by splicng their genes together at one point
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.gene_length - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    #implement crossover as a two point crossover to increase diversity
    def two_point_crossover(self, parent1, parent2):
        point1 = random.randint(0, self.gene_length - 1)
        point2 = random.randint(point1, self.gene_length - 1)
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child

    #mutate a gene by changing a random character in the gene to a random direction
    def mutate(self, gene):
        gene_list = list(gene)
        for i in range(len(gene_list)):
            # Mutation rate increases linearly with the position in the gene
            position_based_mutation_rate = self.mutation_rate (i / len(gene_list))
            if random.random() < position_based_mutation_rate:
                gene_list[i] = random.choice('UDLR')
        return ''.join(gene_list)

    def run_genetic_algorithm(self):
        #self.generate_initial_population()
        self.forced_initial_population(get_gene_by_generation(28,'best_gene_20241205-202703.txt'))
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

        #run training
        for generation in range(self.generations):
            start_time = time.time()

            #set the mutation rate to decrease linearly with the generation (almost like simulated annealing)
            self.mutation_rate = self.starting_mutation_rate - generation * self.muation_rate_decay


            fitness_scores = []
            #simulate games for each gene in the population and record their respective fitnesses
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
            
            print(f'Generation {generation + 1} Best fitness: {best_fitness}, Sim Time: {time.time() - start_time}')

            # Elitism: carry over the best genes to the new population
            num_elites = 3
            #don't start making elites until after the first few generations
            if generation < 10:
                num_elites = 0

            if num_elites == 0:
                elites = []
            else:
                elites = [self.population[i] for i in np.argsort(fitness_scores)[-num_elites:]]

            new_population = []
            new_population.extend(elites)

            #breed and create a new, mutated population
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(fitness_scores)
                #child = self.crossover(parent1, parent2)
                child = self.two_point_crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        
        plt.ioff()
        plt.show()

        return best_gene

    '''
    THIS FUNCTION IS USED TO PROGRESS THE GENETIC ALGORITHM BY TAKING A GOOD GENE AND BUILDING OFF OF IT
    Was only used to see if the path could be refined to get a better score, but it was not successful
    '''

    #take a current good segment from a gene and build off of it to find the best path to go from there
    def genetic_algorithim_progression(self):
        #number of genes that you wan to keep from the initial
        segment = 322
        self.forced_initial_population(get_gene_by_generation(7,'best_gene_20241205-212015.txt'))
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

            #set the mutation rate to decrease linearly with the generation (almost like simulated annealing)
            self.mutation_rate = self.starting_mutation_rate # - generation * self.muation_rate_decay


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
            
            print(f'Generation {generation + 1} Best fitness: {best_fitness}, Sim Time: {time.time() - start_time}')

            # Elitism: carry over the best genes to the new population
            num_elites = 2
            #don't start making elites until after the later generations
            if generation < 10:
                num_elites = 0

            if num_elites == 0:
                elites = []
            else:
                elites = [self.population[i] for i in np.argsort(fitness_scores)[-num_elites:]]

            new_population = []
            new_population.extend(elites)
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(fitness_scores)
                child = self.two_point_crossover(parent1, parent2)
                child = self.mutate(child[segment:])  # Mutate only the part after the segment
                child = best_gene[:segment] + child  # Preserve the first "segment" characters
                new_population.append(child)

            self.population = new_population

        
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
                self.clock.tick(500)
        return game

    #fitness function which takes in a pacman game and uses the end state of the game to determine how well the pacman did
    def fitness(self, game):
        #constants for tuning weight of parameters
        k_score = 0
        #turning off survival time for now, only care about pellets
        k_time = 0.04

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
    print(gen_alg.fitness(gen_alg.run_game(gen_alg.test_gene, is_displaying=True)))

def train():
    gen_alg = Genetics()
    print(gen_alg.run_genetic_algorithm())

def olympics():
    gen_alg = Genetics()
    print(gen_alg.genetic_algorithim_progression())

def test_gene(gen, filename='best_gene.txt'):
    gen_alg = Genetics()
    gene = get_gene_by_generation(gen, filename)
    game = gen_alg.run_game(gene, is_displaying=True)
    print(gen_alg.fitness(game))
    print(f'Moves: {game.move_number}')


def main():
    start_time = time.time()
    
    #test_1()
    #test_2()
    #test_3()
    #train()

    #DEMONSTRATION OF THE GENETIC ALGORTHIM WITH THE BEST GENE I WAS ABLE TO PRODUCE
    test_gene(6,'BEST_RESULT.txt')

    end_time = time.time()


    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
