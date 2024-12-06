

# Pacman Genetic Algorithm

This project implements a genetic algorithm to optimize the gameplay of a Pacman game. The algorithm evolves a population of genes, each representing a sequence of moves, to find the most optimal strategy for Pacman to achieve the highest score.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
- [Genetic Algorithm](#genetic-algorithm)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MAVGators/Pacman_Genetic_Algorithim.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Pacman_Genetic_Algorithim
    ```
3. Install the required dependencies:
    ```sh
    pip install pygame matplotlib numpy
    ```

## Usage

To run the Pacman game with the genetic algorithm, execute the following command:
```sh
python GA_Round_2.py
```

## Classes and Functions

### GA_Round_2.py

#### `Genetics` Class
- `__init__(self)`: Initializes the genetic algorithm parameters and game setup.
- `generate_initial_population(self)`: Generates an initial population of random genes.
- `forced_initial_population(self, gene)`: Sets a forced initial population with a given gene.
- `select_parents(self, fitness_scores)`: Selects two parents based on fitness scores.
- `crossover(self, parent1, parent2)`: Performs single-point crossover between two parents.
- `two_point_crossover(self, parent1, parent2)`: Performs two-point crossover between two parents.
- `mutate(self, gene)`: Mutates a gene by changing a random character.
- `run_genetic_algorithm(self)`: Runs the genetic algorithm for a specified number of generations.
- `genetic_algorithim_progression(self)`: Progresses the genetic algorithm by refining a good gene.
- `run_game(self, gene, is_displaying)`: Runs a game with a given gene and returns the game object.
- `fitness(self, game)`: Calculates the fitness score based on the game outcome.

### pacman.py

#### `Game_Setup` Class
- Initializes the game setup, including screen dimensions and sprite paths.

#### `Stopwatch` Class
- Implements a simple stopwatch for timing events.

#### `SoundEffects` Class
- Loads and manages sound effects for the game.

#### `Graphics` Class
- Loads and scales sprites for Pacman, ghosts, and game elements.

#### `Pacman` Class
- Represents Pacman and handles movement, collisions, and scoring.

#### `Ghost` Class (and subclasses `Blinky`, `Pinky`, `Inky`, `Clyde`)
- Represents ghosts and handles movement, target tile calculation, and modes (chase, scatter, frightened).

#### `Maze` Class
- Represents the game maze and manages elements like walls, pellets, and power pellets.

#### `Game` Class
- Manages the main game loop, including updating game state, handling collisions, and rendering.

#### `Genetic_Game` Class
- Extends the `Game` class to include genetic algorithm-specific functionality.

#### `Player` Class
- Represents the player and manages score and lives.

## Genetic Algorithm

The genetic algorithm optimizes Pacman's gameplay by evolving a population of genes over multiple generations. Each gene represents a sequence of moves (U, D, L, R). The algorithm includes:
- Selection: Choosing parents based on fitness scores.
- Crossover: Combining genes of parents to create offspring.
- Mutation: Introducing random changes to genes to maintain diversity.

The fitness function evaluates the performance of each gene based on Pacman's score, time alive, pellets eaten, and whether Pacman wins the game.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README file with additional details or modifications specific to your project.
