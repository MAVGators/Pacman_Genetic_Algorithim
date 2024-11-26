import pygame as pyg
import sys
from enum import Enum
import random
import time
import os



'''
VERY IMPORTANT HELPER FUNCTION WHICH ALLOWS YOU TO PLAY SOUND FOR PACMAN GAME BY READING THE SOUND FILES FROM THE SAME DIRECTORY AS THE GAME
'''
def open_file_in_same_directory(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    return file_path





#scale between the sprite sheet and the actual display
SCALE = 3
# Screen dimensions
#number of cells in the x and y direction (8x8 pixels in the original image consituttes a cell)
X_CELLS = 28
Y_CELLS = 31


SCREEN_WIDTH = X_CELLS*8*SCALE
SCREEN_HEIGHT = Y_CELLS*8*SCALE+120

#cordinattes to the middle ghost cage as this is where all the ghosts start at in terms of cells
BLINKY_CAGE_CORDS = (13,11)
CAGE_CORDS = (13,13)

#pacman starting position
PACMAN_START = (13, 23)

DISPLAYING = False

# Initialize pygame
pyg.init()

# Load the sprite sheet
sprite_sheet_path = open_file_in_same_directory('Arcade - Pac-Man - General Sprites.png')
#arcade font path
arcade_font_path = open_file_in_same_directory('PressStart2P-vaV7.ttf')

ARCADE_FONT = pyg.font.Font(arcade_font_path, 20)
ARCADE_FONT_LARGE = pyg.font.Font(arcade_font_path, 32)

# Create the screen
screen = pyg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


'''

FIGURE OUT THE TIME STEP 


'''

#helper clas for timing sound effects
class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.pause_time = 0.0
    
    def is_running(self):
        return self.start_time is not None
    
    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise Exception("Stopwatch not started.")

        self.elapsed_time += time.time() - self.start_time
        self.start_time = None
        self.pause_time = 0.0

    def get_elapsed_time(self):
        if self.start_time is not None:
            return time.time() - self.start_time - self.pause_time
        else:
            return None
    
    def pause(self):
        self.pause_time += time.time() - self.start_time
        self.start_time = 0.0

    def unpause(self):
        self.start_time = time.time() - self.pause_time
        self.pause_time = 0.0
    

class SoundEffects:
    eating_channel = pyg.mixer.Channel(0)
    background_channel = pyg.mixer.Channel(1)
    startup_channel = pyg.mixer.Channel(2)
    death_channel = pyg.mixer.Channel(3)
    cherry_channel = pyg.mixer.Channel(4)
    ghost_eating_channel = pyg.mixer.Channel(5)
    
    

    eating_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_waka.wav"))
    background_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_background.wav"))
    startup_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_startup.wav"))
    death_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_death.wav"))
    cherry_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_eatfruit.wav"))
    frightened_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_backgroundfrightened.wav"))
    ghost_eating_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_eatghost.wav"))
    win_effect = pyg.mixer.Sound(open_file_in_same_directory("pacman_win.wav"))




class Graphics:
    def __init__(self, sprite_sheet_path):
        self.sprite_sheet = pyg.image.load(sprite_sheet_path).convert()
        #make all the black transparent to avoid weird box around the sprites
        self.sprite_sheet.set_colorkey((0,0,0))


    def get_sprite(self, x, y, width, height):
        sprite = pyg.Surface((width, height), pyg.SRCALPHA)
        sprite.blit(self.sprite_sheet, (0, 0), (x, y, width, height))
        return sprite

    def scale_sprite(self, sprite):
        width = sprite.get_width() * SCALE
        height = sprite.get_height() * SCALE
        return pyg.transform.scale(sprite, (int(width), int(height)))

    def scale_sprites(self, sprites: dict):
        for key in sprites:
            #if the sprite is a list of sprites scale each sprite in the list individually
            if type(sprites[key]) == list:
                for i in range(len(sprites[key])):
                    sprites[key][i] = self.scale_sprite(sprites[key][i])
                    
            #if the sprite is a single sprite scale it
            else:
                sprites[key] = self.scale_sprite(sprites[key])

        return sprites
    
    #take advantage of the fact that the ghost sprites are in a grid and load all the ghost sprites at once
    def load_ghost_sprites(self):
        #initialize dict
        ghost_sprites = {
            'blinky': None,
            'pinky': None,
            'inky': None,
            'clyde': None,
            'frightened': None,
        }

        #add the main ghost sprites
        for ghost_row in range(4):
            sprites = dict()
            sprite_list = []
            for ghost_col in range(8):
                #get the sprite for each ghost
                sprite = self.get_sprite(456 + ghost_col*16, 64 + ghost_row*16, 16, 16)
                #scale the sprite
                sprite = self.scale_sprite(sprite)
                #add the sprite to the list
                sprite_list.append(sprite)


                #assigne correct direction
                if ghost_col == 1:
                    sprites['right'] = sprite_list
                    #reset list
                    sprite_list = []
                elif ghost_col == 3:
                    sprites['left'] = sprite_list
                    #reset list
                    sprite_list = []

                elif ghost_col == 5:
                    sprites['up'] = sprite_list
                    #reset list
                    sprite_list = []

                elif ghost_col == 7:
                    sprites['down'] = sprite_list
                    #reset list
                    sprite_list = []


            #assign the sprite to the correct ghost
            if ghost_row == 0:
                ghost_sprites['blinky'] = sprites
            elif ghost_row == 1:
                ghost_sprites['pinky'] = sprites
            elif ghost_row == 2:
                ghost_sprites['inky'] = sprites
            elif ghost_row == 3:
                ghost_sprites['clyde'] = sprites

        #load the frightened ghost sprites
        scared_sprites = []
        for i in range(4):
            sprite = self.get_sprite(584 + i*16, 64, 16, 16)
            sprite = self.scale_sprite(sprite)
            scared_sprites.append(sprite)
        ghost_sprites['frightened'] = scared_sprites

        #add the dead ghost sprites to each ghost, this is the floating eyes
        for i in range(4):
            sprite = self.get_sprite(584 + i*16, 80, 16, 16)
            sprite = self.scale_sprite(sprite)
            for ghost in ghost_sprites:
                if ghost != 'frightened':
                    for direction in ghost_sprites[ghost]:
                        ghost_sprites[ghost][direction].append(sprite)



        return ghost_sprites

    def load_pacman_sprites(self):
        pacman_sprites = {
            'right': [self.get_sprite(456, 0, 16, 16),self.get_sprite(472, 0, 16, 16)],
            'left': [self.get_sprite(456, 16, 16, 16),self.get_sprite(472, 16, 16, 16)],
            'up': [self.get_sprite(456, 32, 16, 16),self.get_sprite(472, 32, 16, 16)],
            'down': [self.get_sprite(456, 48, 16, 16),self.get_sprite(472, 48, 16, 16)],
            'start': [self.get_sprite(488, 0, 16, 16)],
            'death': []
        }
        #11 stages of the death animation, take advanathe of the fact that the sprites are in a grid
        for i in range(11):
            pacman_sprites['death'].append(self.get_sprite(504 + i*16, 0, 16, 16))
        return pacman_sprites

    def load_sprites(self):
        #save pacman animation sprites, index 0 is the open mouth and index 1 is the closed mouth
        pacman_sprites = self.load_pacman_sprites()
        ghost_sprites = self.load_ghost_sprites()

        background_sprites = {
            'background': self.get_sprite(228, 0, 225, 250),
            'pellet': self.get_sprite(8, 8, 8, 8),
            'power_pellet': self.get_sprite(8, 24, 8, 8),
            'cherry': self.get_sprite(490, 50, 15, 15),
        }

        # Scale to be more visible in display
        self.scale_sprites(pacman_sprites)
        self.scale_sprites(background_sprites)
        #ghost sprites already scaled in load function

        return pacman_sprites, ghost_sprites, background_sprites



'''
PACMAN CLASSES ALL GO BELOW HERE
'''

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5

#types of elements that can be fgound in the maze
class elements(Enum):
    WALL = '#'
    EMPTY = ' '
    PELLET = 'p'
    POWER_PELLET = 'o'
    PACMAN = 'C'
    GHOST = 'G'
    CHERRY = 'c'

#Moveable objects consist of pacman and the ghosts and are any sprite that can move
class Moveable:
    def __init__(self, position, sprites):
        #tuple of the position of the center of the object in terms of pixels of the unscaled sprite
        #tuple of the position of pacman in terms of cells in the maze
        self.position = position
        #integer between 0-8 representing the subposition of pacman in the cell in terms of pixels from the center of tile he is
        self.subposition = (0,0)
        self.direction = Direction.STOP
        #array of sprites with thier different orientations
        self.sprites = sprites
        #alternate between movement animations
        self.open = False
        #keep track of displaying if frightened
        self.flash = False

    #determine current displacemendt covered in one screen by pacman according to his curret speed
    def displacement(self): 
        speed = 1
        time = 1
        return int(speed*time)

    #check if object can change direction and if so change direction, returns true if can turn and false if cannot
    def change_direction(self, direction, maze):
        #check if object is next to wall and if so do not allow him to change direction
        x, y = self.position
        sub_x, sub_y = self.subposition
        
        #special cases for tunnel
        if x == maze.tunnel_left[0] and y == maze.tunnel_left[1]:
            if  direction == Direction.LEFT or direction == Direction.RIGHT:
                self.direction = direction
                return True
            else:
                return False
        elif x == maze.tunnel_right[0] and y == maze.tunnel_right[1]:
            if  direction == Direction.LEFT or direction == Direction.RIGHT:
                self.direction = direction
                return True
            else:
                return False

        #check if pacman can turn in, if not then buffer that move
        if direction == Direction.UP and sub_x==0 and maze.maze_elems[y-1][x] != elements.WALL:
            self.direction = direction
        elif direction == Direction.DOWN and sub_x==0 and maze.maze_elems[y+1][x] != elements.WALL:
            self.direction = direction
        elif direction == Direction.LEFT and sub_y==0 and maze.maze_elems[y][x-1] != elements.WALL:
            self.direction = direction
        elif direction == Direction.RIGHT and sub_y==0 and maze.maze_elems[y][x+1] != elements.WALL:
            self.direction = direction
        else:
            return False   
        return True
    
    #move the sprite during a frame based on its current direction
    def move(self, maze):
        x, y = self.position
        sub_x, sub_y = self.subposition
        #change orientation of sprite based on where he is trying to move
        direction = self.direction

        #check if the next position is a wall and if not move to that position
        if direction == Direction.UP:
            #at edge of cell move to next cell
            if sub_y == 0 and sub_x == 0 and maze.maze_elems[y - 1][x] != elements.WALL:
                self.position = (x, y - 1)
                #reset sub_y to 15 to move to the bottom of the cell
                self.subposition = (sub_x, 8)
                #change the mouth state for animation
                self.open = not self.open
                #alternate flashing in frightened mode
                self.flash = not self.flash
            else:
                #cannot make sub positio below 0
                self.subposition = (sub_x,max(sub_y - self.displacement(),0))

        elif direction ==  Direction.DOWN and maze.maze_elems[y + 1][x] != elements.WALL:
            #at edge of cell move to next cell
            if sub_y == 8 and sub_x == 0:
                self.position = (x, y + 1)
                #reset sub_y to 0 to move to the top of the cell
                self.subposition = (sub_x, 0)
                #change the mouth state for animation
                self.open = not self.open
                #alternate flashing in frightened mode
                self.flash = not self.flash
            elif sub_x == 0:
                self.subposition = (sub_x, min(sub_y + self.displacement(),8))

        elif direction == Direction.LEFT:
            #special case for tunnel
            if x == maze.tunnel_left[0] and y == maze.tunnel_left[1]:
                if sub_x == 0 and sub_y==0:
                    self.position = maze.tunnel_right
                    self.subposition = (8, sub_y)
                else:
                    self.subposition = (max(sub_x - self.displacement(), 0), sub_y)

            #at edge of cell move to next cell
            elif sub_x == 0 and sub_y==0 and maze.maze_elems[y][x - 1] != elements.WALL:
                self.position = (x - 1, y)
                #reset sub_x to 15 to move to the right of the cell
                self.subposition = (8, sub_y)
                #change the mouth state for animation
                self.open = not self.open
                #alternate flashing in frightened mode
                self.flash = not self.flash
            elif sub_y == 0:
                self.subposition = (max(sub_x - self.displacement(), 0), sub_y)

        elif direction == Direction.RIGHT:
            #special case for tunnel
            if x == maze.tunnel_right[0] and y == maze.tunnel_right[1]:
                if sub_x == 8 and sub_y==0:
                    self.position = maze.tunnel_left
                    self.subposition = (0, sub_y)
                else:
                    self.subposition = (min(sub_x + self.displacement(),8), sub_y)
            
            #at edge of cell move to next cell
            elif sub_x == 8 and sub_y==0 and maze.maze_elems[y][x + 1] != elements.WALL:
                self.position = (x + 1, y)
                #reset sub_x to 0 to move to the left of the cell
                self.subposition = (0, sub_y)
                #change the mouth state for animation
                self.open = not self.open
                #alternate flashing in frightened mode
                self.flash = not self.flash
            elif sub_y == 0 and maze.maze_elems[y][x + 1] != elements.WALL:
                self.subposition = (min(sub_x + self.displacement(),8), sub_y)

    '''
    hellped function to check if the buffered direction is opposite a direction you
    are travelling and if so remove the buffered direction, this is done
    to avoid weird corner glitches as if you are taking a cornner with a floor below you 
    and you have a down move buffered the second you try to go up you will now be able to execute
    the buffered down move meaning you will go up then immediatlly reset your position

    ALso helps in ghosts to avoid them turning around and tracing back their steps
    '''
    def check_opposite_drection(self, new_direction):
        if self.direction == Direction.UP and new_direction == Direction.DOWN:
            new_direction = Direction.STOP
        elif self.direction == Direction.DOWN and new_direction == Direction.UP:
            new_direction = Direction.STOP
        elif self.direction == Direction.LEFT and new_direction == Direction.RIGHT:
            new_direction = Direction.STOP
        elif self.direction == Direction.RIGHT and new_direction == Direction.LEFT:
            new_direction = Direction.STOP
        #return true if the buffered move is opposite the current direction, and false otherwise
        else:
            return False
        return True
    
    def display(self, screen):
        x,y = self.position
        sub_x, sub_y = self.subposition
        
        sprite = None
        #choose sprite based on whether mouth is open or closed and the direction pacman is facing
        if self.direction == Direction.UP:
            if self.open:
                sprite = self.sprites['up'][0]
            else:
                sprite = self.sprites['up'][1]
        elif self.direction == Direction.DOWN:
            if self.open:
                sprite = self.sprites['down'][0]
            else:
                sprite = self.sprites['down'][1]
        elif self.direction == Direction.LEFT:
            if self.open:
                sprite = self.sprites['left'][0]
            else:
                sprite = self.sprites['left'][1]
                
        #start the game facing right in the stop direction
        elif self.direction == Direction.RIGHT or self.direction == Direction.STOP:
            if self.open:
                sprite = self.sprites['right'][0]
            else:
                sprite = self.sprites['right'][1]

        #since position is at the center point of pacman we need to adjust the position to the top left corner of the spritea
        screen.blit(sprite, ((x*8+sub_x-4)*SCALE, (y*8+sub_y-4)*SCALE))

    #check if object runs into another movable object, returns true if there is a collision and false otherwise
    def collision(self, other : 'Ghost'):
        #if ghost is already eaten by pacman do not collide with him
        if self.position == other.position and not other.is_eaten:
            return True
        else: 
            return False
    
class Pacman(Moveable):
    def __init__(self, sprites, player):
        starting_position = PACMAN_START
        #construct a moveable object with the starting position
        super().__init__(starting_position, sprites)

        #change sub position for display 
        self.subposition = (5,0)
        #player who controls pacman
        self.player = player

        #dictionry containing all the different sprites for the different directions
        self.sprites = sprites

        '''
        input buffering, if the player tries to change direction and pacman is not able to change direction at that time
        save the direction the player wants to go in and change direction when pacman is able to
        '''
        self.buffered = Direction.STOP

        #keep track of whether pacman is supercharged py power pellet or not
        self.is_supercharged = False

        #alternate wether mouth is open or closed for animation
        self.open = True

        #timer for determining when to play eating sound effect
        self.eating_timer = Stopwatch()
        self.eat_time = 0.15


    #reset pacman to starting position
    def reset(self):
        self.position = PACMAN_START
        self.direction = Direction.STOP
        self.subposition = (5,0)
        self.buffered = Direction.STOP
        self.is_supercharged = False

    #override inherrented change directiojn to include buffering inputs
    #check to make sure pacman can change to the intended direction and change if able, return true if can turn return false if cannot
    def change_direction(self, direction, maze):
        #check if pacman is next to wall and if so do not allow him to change direction
        x, y = self.position
        sub_x, sub_y = self.subposition

        #prevent buffered moves in the opposite direction from executing, see more information at actual function definition
        if super().check_opposite_drection(self.buffered):
            self.buffered = Direction.STOP
            return False
        
        #spec`ial case for tunnel
        if x == maze.tunnel_left[0] and y == maze.tunnel_left[1]:
            if  direction == Direction.LEFT or direction == Direction.RIGHT:
                self.direction = direction
                return True
            else:
                self.buffered = direction
                return False
        elif x == maze.tunnel_right[0] and y == maze.tunnel_right[1]:
            if  direction == Direction.LEFT or direction == Direction.RIGHT:
                self.direction = direction
                return True
            else:
                self.buffered = direction
                return False

        #check if pacman can turn in, if not then buffer that move
        if direction == Direction.UP and sub_x==0 and maze.maze_elems[y-1][x] != elements.WALL:
            self.direction = direction
        elif direction == Direction.DOWN and sub_x==0 and maze.maze_elems[y+1][x] != elements.WALL:
            self.direction = direction
        elif direction == Direction.LEFT and sub_y==0 and maze.maze_elems[y][x-1] != elements.WALL:
            self.direction = direction
        elif direction == Direction.RIGHT and sub_y==0 and maze.maze_elems[y][x+1] != elements.WALL:
            self.direction = direction
        #can't turn
        else:
            #set buffered direction 
            self.buffered = direction
            return False
        #did turn 
        return True    
    
                
    #slighlty override parent move function to also implement eating pellets
    def move(self, maze):
        super().move(maze)
        #eat pellet if there is one at that position
        self.eat(maze.maze_elems)

    def eat(self, maze):
        x, y = self.position
        #eat the pellet if there is one at the position by incrementing score and removing the pellet
        if maze[y][x] == elements.PELLET:
            maze[y][x] = elements.EMPTY
            self.player.score += 10
            # Play pellet eating sound effect only if not alread playing
            if self.eating_timer.start_time == None or self.eating_timer.get_elapsed_time() == 0:
                #Turn off eating sound REMOVE LATER TO PLAY REGULARLY
                # SoundEffects.eating_channel.play(SoundEffects.eating_effect)
                self.eating_timer.start()
            elif self.eating_timer.get_elapsed_time() >= self.eat_time:
                self.eating_timer.stop()

        #eat the power pellet if there is one at the position by incrementing score and removing the pellet and making pacman supercharged
        if maze[y][x] == elements.POWER_PELLET:
            maze[y][x] = elements.EMPTY
            self.player.score += 50
            self.is_supercharged = True
        
        #eat the cherry for an additional 100 points
        if maze[y][x] == elements.CHERRY:
            #play cherry noise
            #REMOVE CHERRY NOISE ADD LATER
            #SoundEffects.cherry_channel.play(SoundEffects.cherry_effect)
            maze[y][x] = elements.EMPTY
            self.player.score += 100
        

    def display(self, screen):
        super().display(screen)
        #display text
        font = pyg.font.Font(None, 36)
        text = ARCADE_FONT.render(f'Score: {self.player.score}', True, (255, 255, 255))
        screen.blit(text, (10, SCREEN_HEIGHT - 90))
        text = ARCADE_FONT.render(f'Lives: {self.player.lives}', True, (255, 255, 255))
        screen.blit(text, (10, SCREEN_HEIGHT - 30))
        text = ARCADE_FONT.render(f'SC?: {self.is_supercharged}', True, (255, 255, 255))
        screen.blit(text, (250, SCREEN_HEIGHT - 30))
        # text = ARCADE_FONT.render(f'Direction: {self.direction}', True, (255, 255, 255))
        # screen.blit(text, (10, SCREEN_HEIGHT - 30))
        # text = ARCADE_FONT.render(f'Buffered: {self.buffered}', True, (255, 255, 255))
        # screen.blit(text, (350, SCREEN_HEIGHT - 30))



class Ghost(Moveable):
    def __init__(self, color, sprites, frightened_sprites):
        #every ghost except blinky starts in cage
        self.position = CAGE_CORDS
        super().__init__(self.position, sprites)
        self.frightened_sprites = frightened_sprites
        self.sub_position = (0,0)
        self.color = color
        self.sprites = sprites
        self.direction = Direction.STOP

        #every ghost has a target tile they want to reach and, this determiens the direction they will move in
        self.target_tile = (0,0)
        #every ghost has a scatter tile they want to reach when they are in scatter mode
        self.scatter_tile = (0,0)

        self.mode = 'chase'

        self.left_cage = False

        #additional fields for when the ghost is eaten by pacman in frightened mode
        self.is_eaten = False


    
    #reset the ghost to the cage after pacman death
    def reset(self):
        self.position = CAGE_CORDS
        self.direction = Direction.STOP
        self.subposition = (0,0)
        self.mode = 'chase'
        self.left_cage = False

    #move the ghost out of the cage, to start chasign the player
    def leave_cage(self, maze):
        #first naviagate to the center of the cage
        if self.position != CAGE_CORDS:
            self.target_tile = CAGE_CORDS
            self.choose_direction(maze)
            self.move(maze)
        
        #if the ghost is in the middle of the cage forciblly move him out of the cage
        elif self.position == CAGE_CORDS:
            self.position = (self.position[0], self.position[1] - 2)
            self.subposition = (0,0)
            self.direction = Direction.UP
            self.left_cage = True

    #every ghost has a different chase algorithm that will be overriden by the subclass
    #determines the target tile for the ghost to move to
    def chase(self, pacman, graph):
        pass

    def scatter(self):
        self.target_tile = self.scatter_tile
    
    #will determine the direction the ghost will move in to catch pacman
    def choose_direction(self, maze):
        #determine possible directions (neighbors of the current position)
        possible_tiles = maze.graph[self.position]
        #dtermine the possible directions the ghost can move in based on the tiles available to him
        possible_directions = {
            Direction.UP: None,
            Direction.DOWN: None,
            Direction.LEFT: None,
            Direction.RIGHT: None
        }

        #figure out what directions the ghost must move in to get to that tile
        #special tunnel cases
        if self.position == maze.tunnel_left:
            possible_directions[Direction.LEFT] = maze.tunnel_right
        elif self.position == maze.tunnel_right:
            possible_directions[Direction.RIGHT] = maze.tunnel_left

        for tile in possible_tiles:            
            if tile[0] == self.position[0] and tile[1] < self.position[1]:
                possible_directions[Direction.UP] = tile
            elif tile[0] == self.position[0] and tile[1] > self.position[1]:
                possible_directions[Direction.DOWN] = tile
            elif tile[1] == self.position[1] and tile[0] < self.position[0]:
                possible_directions[Direction.LEFT] = tile
            elif tile[1] == self.position[1] and tile[0] > self.position[0]:
                possible_directions[Direction.RIGHT] = tile

        #determine the distance to the target tile for each possible direction
        directions = {
            Direction.UP: None,
            Direction.DOWN: None,
            Direction.LEFT: None,
            Direction.RIGHT: None
        }

        for direction in directions:
            #can't choose to reverse direction or turn somewhere you cannot go
            if possible_directions[direction] is None or super().check_opposite_drection(direction):
                directions[direction] = float('inf')
            else:
                #distance from possible tile to target tile
                directions[direction] = self.distance(possible_directions[direction], self.target_tile)
            #choose lowest cost direction to travel
        #print(directions)
        optimal_direction = min(directions, key=directions.get)
        #if have reached edge of tile to turn that way do so, if not then wait 
        self.change_direction(optimal_direction, maze)
        

    #determine the euclidean distance between two points
    def distance(self, start, end):
        return ((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5
    
    #override display functionallity to include frightened mode
    def display(self, screen, flash):
        #regular display if not frightened
        if self.mode != 'frightened':
            super().display(screen)
        
        #ghost is eaten by pacman, display eyes
        elif self.is_eaten:
            x,y = self.position
            sub_x, sub_y = self.subposition
            
            sprite = None
            #choose sprite based on whether mouth is open or closed and the direction pacman is facing
            if self.direction == Direction.UP:
                sprite = self.sprites['up'][2]
            elif self.direction == Direction.DOWN:
                sprite = self.sprites['down'][2]
            elif self.direction == Direction.LEFT:
                sprite = self.sprites['left'][2]
            #start the game facing right in the stop direction
            elif self.direction == Direction.RIGHT or self.direction == Direction.STOP:
                sprite = self.sprites['right'][2]
            #since position is at the center point of pacman we need to adjust the position to the top left corner of the spritea
            screen.blit(sprite, ((x*8+sub_x-4)*SCALE, (y*8+sub_y-4)*SCALE))
        
        #display frightened mode
        elif self.mode == 'frightened':
            x,y = self.position
            sub_x, sub_y = self.subposition
            sprite = None

            #during regular frightened mode the ghost will alternate between the two blue sprites
            #when the ghost starts flashing the sprite will alternate between the four sprites, 2 flashing sprites and the 2 blue sprites
            if not flash or not self.flash:
                if open:
                    sprite = self.frightened_sprites[0]
                else:
                    sprite = self.frightened_sprites[1]


            elif self.flash and flash:
                if open:
                    sprite = self.frightened_sprites[2]
                else:
                    sprite = self.frightened_sprites[3]

            #since position is at the center point of pacman we need to adjust the position to the top left corner of the spritea
            screen.blit(sprite, ((x*8+sub_x-4)*SCALE, (y*8+sub_y-4)*SCALE))

    #during firghtened mode the ghost will move in the opposite direction
    def reverse_direction(self):
        if self.direction == Direction.UP:
            self.direction = Direction.DOWN
        elif self.direction == Direction.DOWN:
            self.direction = Direction.UP
        elif self.direction == Direction.LEFT:
            self.direction = Direction.RIGHT
        elif self.direction == Direction.RIGHT:
            self.direction = Direction.LEFT

    #enter firghted mode and reverse direction
    def make_frightened(self):
        self.mode = 'frightened'
        self.reverse_direction()

    #randomly choose a direction at interesection to move in when firghtened, if not at intersection keep moving straight
    def frightened(self, maze):
        #determine possible directions (neighbors of the current position)
        possible_tiles = maze.graph[self.position]
        #dtermine the possible directions the ghost can move in based on the tiles available to him
        possible_directions = {
            Direction.UP: None,
            Direction.DOWN: None,
            Direction.LEFT: None,
            Direction.RIGHT: None
        }
        #figure out what directions the ghost must move in to get to that tile

        #special tunnel cases
        if self.position == maze.tunnel_left:
            possible_directions[Direction.LEFT] = maze.tunnel_right
        elif self.position == maze.tunnel_right:
            possible_directions[Direction.RIGHT] = maze.tunnel_left

        for tile in possible_tiles:            
            if tile[0] == self.position[0] and tile[1] < self.position[1]:
                possible_directions[Direction.UP] = tile
            elif tile[0] == self.position[0] and tile[1] > self.position[1]:
                possible_directions[Direction.DOWN] = tile
            elif tile[1] == self.position[1] and tile[0] < self.position[0]:
                possible_directions[Direction.LEFT] = tile
            elif tile[1] == self.position[1] and tile[0] > self.position[0]:
                possible_directions[Direction.RIGHT] = tile

        #choose random direction
        directions = [direction for direction in possible_directions if possible_directions[direction] is not None and not super().check_opposite_drection(direction)]
        #if have reached edge of tile to turn that way do so, if not then wait 
        self.change_direction(random.choice(directions), maze)



    #full movement update of the ghost, selects the direction to move in and then moves
    def update(self, pacman, maze):
        if self.mode == 'chase':
            self.chase(pacman, maze)
            self.choose_direction(maze)
            self.move(maze)
        elif self.mode == 'scatter':
            self.scatter()
            self.choose_direction(maze)
            self.move(maze)
    
        elif self.mode == 'frightened':
            #if the ghost is eaten by pacman, he will move to the front of the cage which is Blinky's starting position
            if self.is_eaten and self.position != BLINKY_CAGE_CORDS:
                self.target_tile = BLINKY_CAGE_CORDS
                self.choose_direction(maze)
                self.move(maze)

            #reached starting point, ghost respawns and is no longer frightened even if pacman is supercharges
            elif self.is_eaten and self.position == BLINKY_CAGE_CORDS:
                self.is_eaten = False
                self.mode = 'chase'

            #normL frightened movement
            else:
                #no target tile in frightened mode just direction
                self.frightened(maze)
                self.move(maze)




class Blinky(Ghost):
    def __init__(self, sprite, frightened_sprites):
        super().__init__('red', sprite, frightened_sprites)
        #blinky starts outside of the cage
        self.position = BLINKY_CAGE_CORDS
        self.left_cage = True
        #many of the ghosts scatter tiles are inaccesible, but this does not affect pathfinding to it 
        self.scatter_tile = 25, -3

    #reset the ghost to the cage after pacman death
    def reset(self):
        self.position = BLINKY_CAGE_CORDS
        self.direction = Direction.STOP
        self.subposition = (0,0)
        self.mode = 'chase'

    def chase(self, pacman, maze):
        # Blinky directly targets Pacman's position
        self.target_tile = pacman.position

class Pinky(Ghost):
    def __init__(self, sprite, frightened_sprites):
        super().__init__('pink', sprite, frightened_sprites)
        #many of the ghosts scatter tiles are inaccesible, but this does not affect pathfinding to it 
        self.scatter_tile = (2,-3)

    def chase(self, pacman, maze):
        # Pinky targets a tile 4 tiles ahead of Pacman's current direction
        if(pacman.direction == Direction.UP):
            self.target_tile = (pacman.position[0], pacman.position[1] - 4)
        elif(pacman.direction == Direction.DOWN):
            self.target_tile = (pacman.position[0], pacman.position[1] + 4)
        elif(pacman.direction == Direction.LEFT):
            self.target_tile = (pacman.position[0] - 4, pacman.position[1])
        elif(pacman.direction == Direction.RIGHT):
            self.target_tile = (pacman.position[0] + 4, pacman.position[1])



class Inky(Ghost):
    def __init__(self, sprite, frightened_sprites):
        super().__init__('cyan', sprite, frightened_sprites)
        self.scatter_tile = (27, 33)

    '''
    Inky uses a combination of Pacman's and Blinky's positions
    first check the tile 2 tiles in front of pacman and then draw a vector from blinky to that tile
    then double the length of that vector the tile that vector ends on is the target tile
    '''
    def chase(self, pacman, maze, blinky):
        initial_target = None
        # Pinky targets a tile 4 tiles ahead of Pacman's current direction
        if(pacman.direction == Direction.UP):
            initial_target = (pacman.position[0], pacman.position[1] - 4)
        elif(pacman.direction == Direction.DOWN):
            initial_target = (pacman.position[0], pacman.position[1] + 4)
        elif(pacman.direction == Direction.LEFT):
            initial_target = (pacman.position[0] - 4, pacman.position[1])
        
        #consider pacman to be facing right if he is not moving
        elif(pacman.direction == Direction.RIGHT or pacman.direction == Direction.STOP):
            initial_target = (pacman.position[0] + 4, pacman.position[1])
        
        #calcualte components of vecotr from blinky to target
        dx = initial_target[0] - blinky.position[0]
        #reverse y component as y increases downwards 
        dy = blinky.position[1] - initial_target[1]

        #double the length of the vector
        self.target_tile = (blinky.position[0] + 2*dx, blinky.position[1] + 2*dy)


    #ovveride update to include blinky as a parameter
    #full movement update of the ghost, selects the direction to move in and then moves
    def update(self, pacman, maze, blinky):
        if self.mode == 'chase':
            self.chase(pacman, maze, blinky)
            self.choose_direction(maze)
            self.move(maze)
        elif self.mode == 'scatter':
            self.scatter()
            self.choose_direction(maze)
            self.move(maze)
    
        elif self.mode == 'frightened':
            #if the ghost is eaten by pacman, he will move to the front of the cage which is Blinky's starting position
            if self.is_eaten and self.position != BLINKY_CAGE_CORDS:
                self.target_tile = BLINKY_CAGE_CORDS
                self.choose_direction(maze)
                self.move(maze)

            #reached starting point, ghost respawns and is no longer frightened even if pacman is supercharges
            elif self.is_eaten and self.position == BLINKY_CAGE_CORDS:
                self.is_eaten = False
                self.mode = 'chase'

            #normL frightened movement
            else:
                #no target tile in frightened mode just direction
                self.frightened(maze)
                self.move(maze)


class Clyde(Ghost):
    def __init__(self,sprite, frightened_sprites):
        super().__init__('orange', sprite, frightened_sprites)
        self.scatter_tile = (0,33)

    def chase(self, pacman, maze):
        # Clyde switches between targeting Pacman and wandering

        #targets like blinky when further than 8 tiles away
        if self.distance(self.position, pacman.position) > 8:
            self.target_tile = pacman.position
        #targets scatter tile when closer than 8 tiles away
        else:
            self.scatter()

    


class Background:
    def __init__(self, sprite):
        self.sprite = sprite



#represents the maze and its walls
class Maze:
    def __init__(self, sprites):
        #load all the sprites for the maze background such as the pellet, power pellet and the background itself
        self.sprites = sprites
        '''
        maze boudary itself is represented by 2d binary array where 1 represents a wall and a 0 represents a movable space type of element
        manually initializing the initial maze
        '''
        self.maze_boundaries = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], #0
            [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1], #1
            [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1], #2
            [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1], #3
            [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1], #4
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], #5
            [1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1], #6
            [1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1], #7
            [1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1], #8
            [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1], #9
            [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1], #10
            [1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1], #11
            [1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1], #12
            [1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,1], #13
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], #14
            [1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,1], #15
            [1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1], #16
            [1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1], #17
            [1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1], #18
            [1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1], #19
            [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1], #20
            [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1], #21
            [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1], #22
            [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1], #23
            [1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1], #24
            [1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1], #25
            [1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1], #26
            [1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1], #27
            [1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1], #28
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], #29
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], #30
        ]

        #keep track of where pellets are on the maze
        self.maze_elems = [[elements.EMPTY for x in range(X_CELLS)] for y in range(Y_CELLS)]
        #start with the pellets in their initial positions
        self.fill_maze()

        #define tunnel points where pacman can pass through
        self.tunnel_points = [(0,14), (27,14)]
        self.tunnel_left = self.tunnel_points[0]
        self.tunnel_right = self.tunnel_points[1]
        
        #graphical representation of the maze for the ghosts path finding
        self.graph = self.construct_graph()

    

    #color all collision objects in the maze
    def debug_display_maze(self, screen):
        for y in range(len(self.maze_boundaries)):
            for x in range(len(self.maze_boundaries[y])):
                if self.maze_boundaries[y][x] == 1:
                    pyg.draw.rect(screen, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), (x*8*SCALE, y*8*SCALE, 8*SCALE, 8*SCALE))
        pyg.display.flip()

    #fill maze with the correct ellements
    def fill_maze(self):
        for y in range(Y_CELLS):
            for x in range(X_CELLS):
                #place wall objects at boundaries 
                if self.maze_boundaries[y][x] == 1:
                    self.maze_elems[y][x] = elements.WALL

                #place power pellets in their 4 spawning positions
                elif (y == 3 or y == 23) and (x == 1 or x == 26):
                    self.maze_elems[y][x] = elements.POWER_PELLET

                #place col of pellets that goes through middle of maze
                elif x == 6 or x == 21:
                    self.maze_elems[y][x] = elements.PELLET
                
                #pellets are placed in every empty slot except for the ghost cage and the tunnel and the middle
                elif y < 9 or y > 19:
                    self.maze_elems[y][x] = elements.PELLET
                else:
                    self.maze_elems[y][x] = elements.EMPTY
    
    #place a cherry on the board
    def place_cherry(self):
        placed = False
        #find a random location to place the 
        while not placed:
            random_x = random.randint(0, X_CELLS - 1)
            random_y = random.randint(0, Y_CELLS - 1)
            #cherry must be placed on an open space, can't be placed inside the cage
            if (not (random_y > 12 and random_y < 16)) and self.maze_elems[random_y][random_x] == elements.EMPTY:
                self.maze_elems[random_y][random_x] = elements.CHERRY
                placed = True
        
    def construct_graph(self):
        graph = {}
        for y in range(Y_CELLS):
            for x in range(X_CELLS):
                if self.maze_boundaries[y][x] == 0:  # Only consider empty spaces
                    neighbors = []
                    #consider the tunnel as a connection between the two points
                    #special tunnel cases
                    if (x, y) == self.tunnel_left:
                        neighbors.append(self.tunnel_right)
                        neighbors.append((x+1,y))
                    elif (x, y) == self.tunnel_right:
                        neighbors.append(self.tunnel_left)
                        neighbors.append((x-1,y))
                    else:
                        if y > 0 and self.maze_boundaries[y - 1][x] == 0:
                            neighbors.append((x, y - 1))
                        if y < Y_CELLS - 1 and self.maze_boundaries[y + 1][x] == 0:
                            neighbors.append((x, y + 1))
                        if x > 0 and self.maze_boundaries[y][x - 1] == 0:
                            neighbors.append((x - 1, y))
                        if x < X_CELLS - 1 and self.maze_boundaries[y][x + 1] == 0:
                            neighbors.append((x + 1, y))
                    

                    graph[(x, y)] = neighbors
        return graph


    #display the maze on the screen
    def display(self, screen):
        #reset the screen
        screen.fill((0, 0, 0))
        #display the background
        screen.blit(self.sprites['background'], (0, 0))
        #add the pixels and power pellets ontop of background
        for y in range(Y_CELLS):
            for x in range(X_CELLS):
                if self.maze_elems[y][x] == elements.PELLET:
                    screen.blit(self.sprites['pellet'], (x*8*SCALE, y*8*SCALE))
                elif self.maze_elems[y][x] == elements.POWER_PELLET:
                    screen.blit(self.sprites['power_pellet'], (x*8*SCALE, y*8*SCALE))
                elif self.maze_elems[y][x] == elements.CHERRY:
                    screen.blit(self.sprites['cherry'], (x*8*SCALE, y*8*SCALE))




class Game:
    def __init__(self, screen, clock, is_displaying):
        self.screen = screen
        #initialize the graphics wiht the sprite sheet containing all images of pacman and ghosts
        graphics = Graphics(sprite_sheet_path)

        #initialize player and game variables
        self.player = Player('Schools Dollar')

        #set up the sprites
        pacman_sprites, ghost_sprites, background_sprites = graphics.load_sprites()
        self.pacman = Pacman(pacman_sprites, self.player)
        self.ghosts = {
            'blinky': Blinky(ghost_sprites['blinky'], ghost_sprites['frightened']),
            'pinky': Pinky(ghost_sprites['pinky'], ghost_sprites['frightened']),
            'inky': Inky(ghost_sprites['inky'],ghost_sprites['frightened']),
            'clyde': Clyde(ghost_sprites['clyde'],ghost_sprites['frightened'])
        }
        self.maze = Maze(background_sprites)

        self.is_displaying = is_displaying

        self.clock = clock

        #used to keep track of when start animation is playing and when to start the game
        self.start_frame = 0
        self.starting_up = True
        #frame on which the starting animation ends
        self.end_frames = 10

        self.death_frame = 0
        self.death_end_frames = 10

        self.is_dead = False

        #keep track of the game timer
        self.game_timer = Stopwatch()
        self.frightened_timer = Stopwatch()
        #how mnay seconds the ghosts will be frightened for
        self.frightened_time = 10

        #keep track of what mode the ghosts are in
        self.mode = 'scatter'

        #keep track of wether a cherry has been placed or not
        self.cherry_placed = False

        #keep track of when game ends
        self.game_over_bool = False

    #play starting animation and begin game loop
    def start(self):
        #play startup sound starting on first frame
        if(self.start_frame == 0):
            #reset ghost location 
            for ghost in self.ghosts.values():
                ghost.reset()
            #reset pacman location
            self.pacman.reset()

            #only play sound on first round, not on other resets
            if self.player.lives == 3 and self.is_displaying:
                sound = pyg.mixer.Sound(SoundEffects.startup_effect)   
                sound.play()

        if self.is_displaying:
            #play starting animation (which is just pacman waiting for a few seconds)
            self.maze.display(self.screen)
            #since position is at the center point of pacman we need to adjust the position to the top left corner of the spritea
            sprite = self.pacman.sprites['start'][0]
            x,y = self.pacman.position
            sub_x, sub_y = self.pacman.subposition
            screen.blit(sprite, ((x*8+sub_x-4)*SCALE, (y*8+sub_y-4)*SCALE))        
            for ghost in self.ghosts.values():
                ghost.display(self.screen, flash = False)
                
            #render ready text on middle of screen
            text = ARCADE_FONT_LARGE.render('READY?', True, (255, 255, 255))
            self.screen.blit(text, (SCREEN_WIDTH/2.75,SCREEN_HEIGHT/2.1))

        #increment frame counter
        self.start_frame += 1

        #last frame of startup
        if(self.start_frame == self.end_frames - 1):
            #start the background music right as startup ends
            if self.is_displaying:
                SoundEffects.background_channel.play(SoundEffects.background_effect, loops = -1)
            #start game timer
            self.game_timer.start()
            self.starting_up = False
    
    #play the death animation and reset the game
    def reset_round(self):
        #play death noise
        if self.death_frame == 0:
            #reset timer
            self.game_timer.stop()
            if self.is_displaying:
                #cut all other noises
                SoundEffects.background_channel.stop()
                SoundEffects.eating_channel.stop()
                SoundEffects.startup_channel.stop()

                SoundEffects.death_channel.play(SoundEffects.death_effect)
        #play death animation
        #Hardcode death animation
        death_sprite = None
        if self.death_frame < self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][0]
        elif self.death_frame < 2*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][1]
        elif self.death_frame < 3*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][2]
        elif self.death_frame < 4*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][3]
        elif self.death_frame < 5*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][4]
        elif self.death_frame < 6*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][5]
        elif self.death_frame < 7*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][6]
        elif self.death_frame < 8*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][7]
        elif self.death_frame < 9*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][8]
        elif self.death_frame < 10*self.death_end_frames/11:
            death_sprite = self.pacman.sprites['death'][9]
        else:
            death_sprite = self.pacman.sprites['death'][10]

        #increment frame counter
        self.death_frame += 1

        #play death animation
        self.maze.display(self.screen)
        #since position is at the center point of pacman we need to adjust the position to the top left corner of the spritea
        x,y = self.pacman.position
        sub_x, sub_y = self.pacman.subposition
        screen.blit(death_sprite, ((x*8+sub_x-4)*SCALE, (y*8+sub_y-4)*SCALE))        

        #end the reset
        if (self.death_frame == self.death_end_frames-1):
            self.is_dead = False
            self.death_frame = 0
            self.start_frame = 0
            #this will cause the game to run the start animation again and reset the board
            self.starting_up = True
            #reset ghost and pacman positions
            self.pacman.position = PACMAN_START

    def update(self):
        
        #death loop
        if self.is_dead and self.death_frame < self.death_end_frames:
            self.reset_round()
            return
        
        #keep game completley over, place here to allow final death animation to play
        elif self.player.lives == 0:
            self.game_over()
            return
        
        elif self.check_win():
            self.win()
        
        #play starting animation and begin game loop
        elif self.start_frame < self.end_frames:
            self.start()
            return


        #game loop

        #determine the mode of the ghosts based on the game timer, when not in frightened mode
        if not self.mode == 'frightened':
            self.choose_mode()
        #attempt to place the cherry on the board
        self.place_cherry()

        # Update based on player inputs
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                pyg.quit()
                sys.exit()
            if event.type == pyg.KEYDOWN:
                if event.key == pyg.K_w:
                    self.pacman.change_direction(direction=Direction.UP, maze=self.maze)
                elif event.key == pyg.K_s:
                    self.pacman.change_direction(direction=Direction.DOWN, maze=self.maze)
                elif event.key == pyg.K_a:
                    self.pacman.change_direction(direction=Direction.LEFT, maze=self.maze)
                elif event.key == pyg.K_d:
                    self.pacman.change_direction(direction=Direction.RIGHT, maze=self.maze)

        #attempt to execute the buffered direction if there is one
        self.pacman.change_direction(self.pacman.buffered, self.maze)
        self.pacman.move(self.maze)
        #check if pacman has eaten a power pellet
        self.check_frightened()

        self.move_ghosts()

        #check for collisions
        self.handle_collisions()

        #dont draw display if game ended
        if(self.game_over_bool != True):
            self.draw_screen()

    #check if all pellets, powerp, and cherry's have been eaten, if so return true and end game with win
    def check_win(self):
        for y in range(Y_CELLS):
            for x in range(X_CELLS):
                if self.maze.maze_elems[y][x] == elements.PELLET or self.maze.maze_elems[y][x] == elements.POWER_PELLET or self.maze.maze_elems[y][x] == elements.CHERRY:
                    return False
        return True

    #check if pacman has eaten a power pellet, and enter firghtened mode if he has
    def check_frightened(self):
        #check if pacman has eaten a power pellet, start timer if so, also extends the timer if pacman eats another power pellet
        if self.pacman.is_supercharged:
            #pause normal game timer
            self.game_timer.pause()
            self.mode = 'frightened'
            self.change_mode()
            #start the timer for frightened mode
            self.frightened_timer.start()
            self.pacman.is_supercharged = False
            #start playing the scary music
            if self.is_displaying:
                SoundEffects.background_channel.stop()
                SoundEffects.background_channel.play(SoundEffects.frightened_effect, loops = -1)

        #supercharged mode lasts 10 seconds
        elif self.mode == 'frightened' and self.frightened_timer.get_elapsed_time() > self.frightened_time:
            self.choose_mode()
            self.frightened_timer.stop()
            #start playing normal background music
            if self.is_displaying:
                SoundEffects.background_channel.stop()
                SoundEffects.background_channel.play(SoundEffects.background_effect, loops = -1)       
            #unpause game timer
            self.game_timer.unpause()    
    
    #every frame there is a chance to place a very small chance to place cherry on the board
    def place_cherry(self):
        #on average it should take 10 seconds to place a cherry
        if random.randint(0,600) == 42 and not self.cherry_placed:
            #place cherry in random location
            self.maze.place_cherry()
            self.cherry_placed = True

    def handle_collisions(self):
        for ghost in self.ghosts.values():
            #collision with pacman in normal state kills pacman
            if ghost.mode != 'frightened' and self.pacman.collision(ghost):
                self.lose_life()
                self.is_dead = True
            #in frightened mode collision with ghost kills ghost
            elif ghost.mode == 'frightened' and self.pacman.collision(ghost):
                ghost.is_eaten = True
                self.player.update_score(200)
                #show eaten
                self.draw_screen()
                #pause the game for 1 seconds
                self.game_timer.pause()
                self.frightened_timer.pause()
                #cut all other noises
                if self.is_displaying:
                    SoundEffects.background_channel.stop()
                    #play ghost eating noise
                    SoundEffects.ghost_eating_channel.play(SoundEffects.ghost_eating_effect)
                #wait for 1 seconds
                #time.sleep(1)
                #unpause the game
                self.game_timer.unpause()
                self.frightened_timer.unpause()
                if self.is_displaying:
                    #start playing normal background music
                    SoundEffects.background_channel.play(SoundEffects.frightened_effect, loops = -1)

    #determine the mode of the ghosts based on the game timer
    def choose_mode(self):
        #0-7 seconds scatter
        if self.game_timer.get_elapsed_time() < 7:
            self.mode = 'scatter'
            self.change_mode()
        #7-27 seconds chase
        elif self.game_timer.get_elapsed_time() < 27:
            self.mode = 'chase'
            self.change_mode()

        #27-34 seconds scatter
        elif self.game_timer.get_elapsed_time() < 34:
            self.mode = 'scatter'
            self.change_mode()

        #34-54 seconds chase
        elif self.game_timer.get_elapsed_time() < 54:
            self.mode = 'chase'
            self.change_mode()
        #54-59 seconds scatter
        elif self.game_timer.get_elapsed_time() < 59:
            self.mode = 'scatter'
            self.change_mode()
        #59-79 seconds chase
        elif self.game_timer.get_elapsed_time() < 79:
            self.mode = 'chase'
            self.change_mode()
        #79-84 seconds scatter
        elif self.game_timer.get_elapsed_time() < 84:
            self.mode = 'scatter'
            self.change_mode()
        #Chase indefinitely
        else:
            self.mode = 'chase'
            self.change_mode()
        
    #change the mode of all ghosts on the sc
    def change_mode(self):
        for ghost in self.ghosts.values():
            ghost.mode = self.mode

    #update the position of the ghosts and also keep track of releasing them from the cage
    def move_ghosts(self):
        self.ghosts['blinky'].update(self.pacman, self.maze)

        #slowly release ghosts one by one from cage
        if self.game_timer.get_elapsed_time() > 7 and not self.ghosts['pinky'].left_cage:
            self.ghosts['pinky'].leave_cage(self.maze)
        else:
            self.ghosts['pinky'].update(self.pacman, self.maze)

        if self.game_timer.get_elapsed_time() > 10 and not self.ghosts['inky'].left_cage:
            self.ghosts['inky'].leave_cage(self.maze)
        else:
            self.ghosts['inky'].update(self.pacman, self.maze, self.ghosts['blinky'])

        if self.game_timer.get_elapsed_time() > 13 and not self.ghosts['clyde'].left_cage:
            self.ghosts['clyde'].leave_cage(self.maze)
        else:        
            self.ghosts['clyde'].update(self.pacman, self.maze)
        

    #occurs whenever pacman dies, reset the round if he has lives and reset the game if he does not
    def lose_life(self):
        self.player.lives -= 1
        self.reset_round()

    def game_over(self):
        #fill screen with black and display text sayinng you lose
        self.screen.fill((0,0,0))
        text = ARCADE_FONT_LARGE.render('GAME OVER', True, (255, 255, 255))
        self.screen.blit(text, (SCREEN_WIDTH/3,SCREEN_HEIGHT/2))
        #display final score
        text = ARCADE_FONT.render(f'Final Score: {self.player.score}', True, (255, 255, 255))
        self.screen.blit(text, (SCREEN_WIDTH/3,SCREEN_HEIGHT/1.5))
        self.game_over_bool = True
        #cut all noise
        SoundEffects.background_channel.stop()
    
    #display win screen and final points
    def win(self):
        #fill screen with black and display text sayinng you lose
        self.screen.fill((255,255,255))
        text = ARCADE_FONT_LARGE.render('YOU WIN', True, (0, 0, 0))
        self.screen.blit(text, (SCREEN_WIDTH/3,SCREEN_HEIGHT/2))
        #display final score
        text = ARCADE_FONT.render(f'Final Score: {self.player.score}', True, (0, 0, 0))
        self.screen.blit(text, (SCREEN_WIDTH/3,SCREEN_HEIGHT/1.5))
        #right as win start the win music
        if not self.game_over_bool and self.is_displaying:
            #cut all noise
            SoundEffects.background_channel.stop()
            #play win noise
            SoundEffects.background_channel.play(SoundEffects.win_effect, loops=-1)
        self.game_over_bool = True


    def draw_screen(self):
        self.maze.display(self.screen)
        self.pacman.display(self.screen)
        if self.mode == "frightened":
            #display regular blue when firghtened
            if self.frightened_timer.get_elapsed_time() < self.frightened_time - 5:
                for ghost in self.ghosts.values():
                    ghost.display(self.screen, flash = False)
            #flash for the last 5 seconds of frightened mode
            else:
                for ghost in self.ghosts.values():
                    ghost.display(self.screen, flash = True)
        else:
            for ghost in self.ghosts.values():
                ghost.display(self.screen, flash = False)


        #show cherry in bottom right if hasn't been placed
        if not self.cherry_placed:
            self.screen.blit(self.maze.sprites['cherry'], (SCREEN_WIDTH - 2*8*SCALE, SCREEN_HEIGHT - 2*8*SCALE))
        #debug
        font = pyg.font.Font(None, 36)
        text = ARCADE_FONT.render(f'FPS: {self.clock.get_fps()}', True, (255, 255, 255))
        self.screen.blit(text, (450,10))
        #text = ARCADE_FONT.render(f'Mode: {self.mode}', True, (255, 255, 255))
        #self.screen.blit(text, (250, SCREEN_HEIGHT - 30))

class Genetic_Game(Game):

    def __init__(self, screen, clock, gene, is_displaying):
        super().__init__(screen, clock, is_displaying)
        self.gene = gene.upper()
        self.genetic_moves = iter(self.gene)
                #map of moves to directions
        self.direction_map = {
            'L': Direction.LEFT,
            'R': Direction.RIGHT,
            'U': Direction.UP,
            'D': Direction.DOWN
        }

        '''
        replacement for the ingame timers is to keep track of ticks
        if we assume we want this game to display on a 60fps screen when it is not being trained then 
        there should be 60 ticks per second meaning each tick is 1/60 of a second or approximately 16.6666666667 milliseconds
        '''
        self.ticks = {
            'game': 0,
            'frightened': 0
        }

        #used to pause flow of game when in interupting mode like frightened or death
        self.game_ticks_paused = False
        #keep track of when in frightened mode
        self.frightened_ticks_paused = True

    #helper functions to convert between ticks and second, conversion is each tick is 1/60 of a second
    def ticks_to_seconds(self, ticks):
        return ticks/60
    def seconds_to_ticks(self, seconds):
        return seconds*60
    

    #Execute the gene by reading through the chromome and returning the next move in the sequence
    def choose_gene_move(self):
        next_move = next(self.genetic_moves, 'N')
        if next_move == 'N':
            return self.last_move
        self.last_move = next_move
        return self.direction_map[next_move]

    def start(self):
        #reset ghost location 
        for ghost in self.ghosts.values():
            ghost.reset()
        #reset pacman location
        self.pacman.reset()
        #start game timer
        self.game_timer.start()
        self.starting_up = False

    def update(self):
        
        #death loop
        if self.is_dead and self.death_frame < self.death_end_frames:
            self.reset_round()
            return
        
        #keep game completley over, place here to allow final death animation to play
        elif self.player.lives == 0:
            self.game_over()
            return
        
        elif self.check_win():
            self.win()

        #play starting animation and begin game loop
        elif self.starting_up:
            self.start()
            return

        #game loop

        #determine the mode of the ghosts based on the game timer, when not in frightened mode
        if not self.mode == 'frightened':
            self.choose_mode()
        #attempt to place the cherry on the board
        self.place_cherry()

        gene_move_direction = self.choose_gene_move()
        Pacman.change_direction(self.pacman, gene_move_direction, self.maze)


        #attempt to execute the buffered direction if there is one
        self.pacman.change_direction(self.pacman.buffered, self.maze)
        self.pacman.move(self.maze)

        #check if pacman has eaten a power pellet
        self.check_frightened()

        self.move_ghosts()

        #check for collisions
        self.handle_collisions()

        #update the game ticks
        if not self.game_ticks_paused:
            self.ticks['game'] += 1
        if not self.frightened_ticks_paused:
            self.ticks['frightened'] += 1

        #dont draw display if game ended
        if(self.game_over_bool != True and self.is_displaying):
            self.draw_screen()
    
    #overide the basic choose mode logic to function based on ticks instead of an actual timer
    def choose_mode(self):
        #0-7 seconds scatter
        if self.ticks['game'] < self.seconds_to_ticks(7):
            self.mode = 'scatter'
            self.change_mode()
        #7-27 seconds chase
        elif self.ticks['game'] < self.seconds_to_ticks(27):
            self.mode = 'chase'
            self.change_mode()

        #27-34 seconds scatter
        elif self.ticks['game'] < self.seconds_to_ticks(34):
            self.mode = 'scatter'
            self.change_mode()

        #34-54 seconds chase
        elif self.ticks['game'] < self.seconds_to_ticks(54):
            self.mode = 'chase'
            self.change_mode()
        #54-59 seconds scatter
        elif self.ticks['game'] < self.seconds_to_ticks(59):
            self.mode = 'scatter'
            self.change_mode()
        #59-79 seconds chase
        elif self.ticks['game'] < self.seconds_to_ticks(79):
            self.mode = 'chase'
            self.change_mode()
        #79-84 seconds scatter
        elif self.ticks['game'] < self.seconds_to_ticks(84):
            self.mode = 'scatter'
            self.change_mode()
        #Chase indefinitely
        else:
            self.mode = 'chase'
            self.change_mode()

    #overide the basic check frightened logic to function based on ticks instead of an actual timer
    #update the position of the ghosts and also keep track of releasing them from the cage
    def move_ghosts(self):
        self.ghosts['blinky'].update(self.pacman, self.maze)

        #slowly release ghosts one by one from cage
        if self.ticks['game'] > self.seconds_to_ticks(7) and not self.ghosts['pinky'].left_cage:
            self.ghosts['pinky'].leave_cage(self.maze)
        else:
            self.ghosts['pinky'].update(self.pacman, self.maze)

        if self.ticks['game'] > self.seconds_to_ticks(10) and not self.ghosts['inky'].left_cage:
            self.ghosts['inky'].leave_cage(self.maze)
        else:
            self.ghosts['inky'].update(self.pacman, self.maze, self.ghosts['blinky'])

        if self.ticks['game'] > self.seconds_to_ticks(13) and not self.ghosts['clyde'].left_cage:
            self.ghosts['clyde'].leave_cage(self.maze)
        else:        
            self.ghosts['clyde'].update(self.pacman, self.maze)
    
    #overide the basic check frightened logic to function based on ticks instead of an actual timer
    #check if pacman has eaten a power pellet, and enter firghtened mode if he has
    def check_frightened(self):
        #check if pacman has eaten a power pellet, start timer if so, also extends the timer if pacman eats another power pellet
        if self.pacman.is_supercharged:
            #pause normal game timer
            self.game_ticks_paused = True
            self.mode = 'frightened'
            self.change_mode()
            #start the timer for frightened mode
            self.frightened_ticks_paused = False

            self.pacman.is_supercharged = False
            #start playing the scary music
            if self.is_displaying:
                SoundEffects.background_channel.stop()
                SoundEffects.background_channel.play(SoundEffects.frightened_effect, loops = -1)

        #supercharged mode lasts 10 seconds
        elif self.mode == 'frightened' and self.ticks['frightened'] > self.seconds_to_ticks(self.frightened_time):
            self.choose_mode()

            #reset frightened mode
            self.ticks['frightened'] = 0
            self.frightened_ticks_paused = True

            #start playing normal background music
            if self.is_displaying:
                SoundEffects.background_channel.stop()
                SoundEffects.background_channel.play(SoundEffects.background_effect, loops = -1)       
            #unpause game timer
            self.game_ticks_paused = False
    
    #overide the basic screen drawing logic to function based on ticks instead of an actual timer
    def draw_screen(self):
        self.maze.display(self.screen)
        self.pacman.display(self.screen)
        if self.mode == "frightened":
            #display regular blue when firghtened
            if self.ticks['frightened'] < self.seconds_to_ticks(self.frightened_time - 5):
                for ghost in self.ghosts.values():
                    ghost.display(self.screen, flash = False)
            #flash for the last 5 seconds of frightened mode
            else:
                for ghost in self.ghosts.values():
                    ghost.display(self.screen, flash = True)
        else:
            for ghost in self.ghosts.values():
                ghost.display(self.screen, flash = False)


        #show cherry in bottom right if hasn't been placed
        if not self.cherry_placed:
            self.screen.blit(self.maze.sprites['cherry'], (SCREEN_WIDTH - 2*8*SCALE, SCREEN_HEIGHT - 2*8*SCALE))
        #debug
        font = pyg.font.Font(None, 36)
        text = ARCADE_FONT.render(f'FPS: {self.clock.get_fps()}', True, (255, 255, 255))
        self.screen.blit(text, (450,10))
        #text = ARCADE_FONT.render(f'Mode: {self.mode}', True, (255, 255, 255))
        #self.screen.blit(text, (250, SCREEN_HEIGHT - 30))


class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.lives = 1

    def update_score(self, points):
        pass

    def lose_life(self):
        pass



def main():
    # Create the game
    clock = pyg.time.Clock()
    game = Game(screen, clock, True)

    # Main game loop
    while True:

        # Update the game
        game.update()

        #update the display
        pyg.display.flip()

        #limit frame rate 
        clock.tick(60)

#class to run a full game of pacman, used in other files
class Pacman_Game():
    def test_game(self, frame_rate):
        # Create the game
        clock = pyg.time.Clock()
        game = Game(screen, clock, True)

        # Main game loop
        while True:

            # Update the game
            game.update()

            #update the display
            pyg.display.flip()

            #limit frame rate 
            clock.tick(frame_rate)
    
    def execute_moves(self, move_string):
        #map of moves to directions
        direction_map = {
            'L': Direction.LEFT,
            'R': Direction.RIGHT,
            'U': Direction.UP,
            'D': Direction.DOWN
        }

        #index of current move in string
        moves = move_string.upper()
        print(moves)
        move_iter = iter(moves)

        # Create the game
        game = Game(screen, pyg.time.Clock(), False)
        while True:
            try:
                move = next(move_iter)
            except StopIteration:
                break

    def genetic_test(self, gene):
        # Create the game
        clock = pyg.time.Clock()
        best_score = 0
        is_displaying = False
        for i in range(1):
            game = Genetic_Game(screen, clock, gene, is_displaying)
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

        

if __name__ == "__main__":
    main()