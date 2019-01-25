from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty, StringProperty, ObjectProperty, NumericProperty, ReferenceListProperty, DictProperty, BooleanProperty
from kivy.lang import Builder
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics import *
from genetic_algorithm import *
from random import randint
from random import randrange
from copy import deepcopy
import numpy as np
import math
from math import acos
from math import sqrt
from math import pi


# TODO:
'''
correlate decay rate in accordance to size, the smaller you are the slower you decay
correlate health to the size
introduce poison resistance 
introduce health from food parameter
'''


#loads in the Kivy file
Builder.load_file('fish.kv')




# Global Constants for the Radii of the food and the poison
POISON_RADIUS = 10
FOOD_RADIUS = 10

# Global Constant for the number of fishes
NUM_FISHES = 9

# Function to compute Distance between two points (coordinates)
def compute_distance(center_point, dest_point):
    distance = np.sqrt(np.square(dest_point[0]-center_point[0]) + np.square(dest_point[1]-center_point[1]))
    return distance

def normalize_magnitude(vector):
    magnitude = vector[0]**2 + vector[1]**2
    magnitude = magnitude**0.5
    if magnitude != 0:
        vector = Vector(vector)*(1/magnitude)
    return vector

# Class for the Fish
class Fish(Widget):

    # Color values designated for health
    colors = ListProperty([[255/255, 0, 0, 1],[255/255, 43/255, 0, 1],[255/255, 85/255, 0, 1],[255/255, 128/255, 0, 1],[255/255, 170/255, 0, 1],[255/255, 213/255, 0, 1],[255/255, 255/255, 0, 1],[213/255, 255/255, 0, 1],[170/255, 255/255, 0, 1],[128/255, 255/255, 0, 1],[85/255, 255/255, 0, 1],[43/255, 255/255, 0, 1],[0, 255/255, 0, 1]])
    color_val = ListProperty([0, 255/255, 0, 1])
    color_index = NumericProperty(12)

    # You'll have to change these later to be dynamic
    health = NumericProperty(0)
    colorLowBound = NumericProperty(0)


    # Variables designated for movement (not to be determined from the DNA)
    angle_val = NumericProperty(1)
    angle=NumericProperty(0)                  
    rotation=NumericProperty(0)
    max_velocity = NumericProperty(2)
    max_force = NumericProperty(0.5)         
    velocity_x=NumericProperty(0) 
    velocity_y=NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    acceleration_x = NumericProperty(0)
    acceleration_y = NumericProperty(0)
    acceleration = ReferenceListProperty(acceleration_x, acceleration_y)
    

    # Variables to be determined by the DNA when implemented
    poison_rad = NumericProperty(0) 
    food_rad = NumericProperty(0)
    size_val = NumericProperty(0)
    food_propensity = NumericProperty(0)
    poison_propensity = NumericProperty(0)

    # objective fitness
    fitness = NumericProperty(0) 

    # DNA once the variables above have been initialized
    dna = StringProperty('')

    # Function to initialize the poison perception radius, food perception radius and the size of the fish (used for the first randomization)
    def initialize_beginning(self):

        # initializing random values for the health, poison_rad, food_rad, size, and the food and poison propensity
        self.health = randint(20,200)
        self.colorLowBound = self.health - self.health/13
        self.poison_rad = randint(1,127)
        self.food_rad = randint(1,127)
        self.size_val = randint(1, 63)
        food_prop = randint(1, 15)
        poison_prop = randint(1, 15)
        self.food_propensity = 1/food_prop
        self.poison_propensity = -1/poison_prop

        attributes = [self.health, self.poison_rad, self.food_rad, self.size_val, food_prop, poison_prop]

        # Generates the DNA string from the initialized values
        final_dna_string = ""
        for i in range(len(attributes)):
            dna_bitstring_val = ""
            if i == 0:
                dna_bitstring_val = '{0:08b}'.format(attributes[i])
            elif i == 1 or i == 2:
                dna_bitstring_val = '{0:07b}'.format(attributes[i])
            elif i == 3:
                dna_bitstring_val = '{0:06b}'.format(attributes[i])
            else:
                if attributes[i] < 0:
                    attributes[i] *= -1
                    dna_bitstring_val = "1" +'{0:04b}'.format(attributes[i])
                else:
                    dna_bitstring_val = "0" + '{0:04b}'.format(attributes[i])

            final_dna_string += dna_bitstring_val

        self.dna = final_dna_string


    # TODO
    def initialize_from_DNA(self, dna_list):

        # initializing the health
        if dna_list[0] == "00000000":
            self.health = 1
            dna_list[0] = "00000001"
        else:
            self.health = int(dna_list[0], 2)

        # Initializing the lower color bound
        self.colorLowBound = self.health - self.health/13

        # Initializing the poison radius
        if dna_list[1] == "0000000":
            self.poison_rad = 1
            dna_list[1] = "0000001"
        else:
            self.poison_rad = int(dna_list[1], 2)

        # Initializing the food radius
        if dna_list[2] == "0000000":
            self.food_rad = 1
            dna_list[2] = "0000001"
        else:
            self.food_rad = int(dna_list[2], 2)

        # Initializing the size
        if dna_list[3] == "000000":
            self.size_val = 1
            dna_list[3] = "000001"
        else:
            self.size_val = int(dna_list[3], 2)

        # Initializing the food perception
        if dna_list[4][0] == 1:
            if dna_list[4][1:] == "0000":
                self.food_propensity = -1
                dna_list[4] = "10001"
            else:
                self.food_propensity = -1/int(dna_list[4][1:], 2)
        else:
            if dna_list[4][1:] == "0000":
                self.food_propensity = 1
                dna_list[4] = "00001"
            else:
                self.food_propensity = 1/int(dna_list[4][1:], 2)

        # Initializing the poison perception
        if dna_list[5][0] == 1:
            if dna_list[5][1:] == "0000":
                self.poison_propensity = -1
                dna_list[5] = "10001"
            else:
                self.poison_propensity = -1/int(dna_list[5][1:], 2)
        else:
            if dna_list[5][1:] == "0000":
                self.poison_propensity= 1
                dna_list[5] = "00001"
            else:
                self.poison_propensity = 1/int(dna_list[5][1:], 2)

        for attr in dna_list:
            self.dna += attr


    # adjusts the steering force based on the target position of the food or the poison
    def seek(self, current_pos,target_pos):
        desired_velocity = Vector(target_pos) - Vector(current_pos)
        desired_velocity = normalize_magnitude(desired_velocity)*self.max_velocity
        steering_force = desired_velocity - Vector(self.velocity)
        steering_force = normalize_magnitude(steering_force)*self.max_force
        return(steering_force)

    # Applies the force to the acceleration
    def applyForce(self, force):
        # print(force)
        self.acceleration = Vector(self.acceleration) + force

    # Function to move the car
    def move(self):
        # print(self.velocity)
        self.velocity = Vector(self.velocity) + self.acceleration
        self.velocity = Vector(normalize_magnitude(self.velocity))*self.max_velocity
        self.pos = Vector(*self.velocity) + self.pos
        self.acceleration = Vector(self.acceleration) * 0

        # Adjusts the color as the health of the car decreases
        if self.health < self.colorLowBound and self.color_index>=0:
            self.color_val = self.colors[self.color_index]
            self.color_index -=1
            self.colorLowBound -= self.health/13

# Class designed for the car's poison radius, this is needed for displying the poison radius of the fish on the gui
class PoisonRadius(Widget):

    # the Center x and y position as well as the radius value
    cx = NumericProperty(0)
    cy = NumericProperty(0)
    radius = NumericProperty(0)

    # Function to initialize the the properties
    def initialize(self, x,y, rad):
        self.cx = x
        self.cy = y
        self.radius = rad

    # Function to move the Poison Radius
    def move(self, x,y):
        self.cx = x
        self.cy = y


# Class designed for the car's food radius, this is needed for displying the poison radius of the fish on the gui
class FoodRadius(Widget):

    # the Center x and y position as well as the radius value
    cx = NumericProperty(0)
    cy = NumericProperty(0)
    radius = NumericProperty(0)

    # Function to initialize the the properties
    def initialize(self, x,y, rad):
        self.cx = x
        self.cy = y
        self.radius = rad

    # Function to move the Food Radius
    def move(self, x,y):
        self.cx = x
        self.cy = y


''' The two classes below are placeholders that will be inherently invoked through the kivy file '''
class Food(Widget):
    pass

class Poison(Widget):
    pass



#This class will be responsible for actually running the simulation itself
class FishSimulation(Widget):

    # This will maintain the running list of the fishes
    fish_list = ListProperty([])

    # This holds the fishes when initialized so the genetic algorithm can run
    fish_list_genetic_pool = ListProperty([])

    # Lists to store the food and the poison
    food_list = ListProperty([])
    poison_list = ListProperty([])

    # This is a dictionary to store the coordinates for the food and the poison
    food_dict = DictProperty({})
    poison_dict = DictProperty({})

    # Contains the label for th current generation
    gen_val = StringProperty("Generation 1")
    gen_count = NumericProperty(1)

    # Count variable that is useful for debugging
    count = NumericProperty(0)

    # Boolean Variable to designate whether it is time to start a new genetic algorithm
    start_new_gen = BooleanProperty(False)

    # Average Fitness on each run
    avg_fitness = NumericProperty(0)

    # This function will populate the fish and the food onto the screen for the ver first time
    def populate_fishes_and_food(self):
        ''' Debugging to print the height and the width of the  actual screen'''
        # print(self.height)
        # print(self.width)
        for i in range(NUM_FISHES):
            fish = Fish()
            fish.initialize_beginning()
            # print(fish.dna)
            # print(fish.health)
            poison_radius=PoisonRadius()
            food_radius=FoodRadius()
            poison_radius.initialize(fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)
            self.fish_list_genetic_pool.append(fish)
            # print(fish.color_val)
        for i in range(int(NUM_FISHES*5)):
            food = Food()
            poison = Poison()

            food_x, food_y = randint(0, 800), randint(0, 600)
            poison_x, poison_y = randint(0, 800), randint(0, 600)

            while poison_x == food_x and poison_y == food_y:
                poison_x, poison_y = randint(0, 800), randint(0, 600)

            food_coord = food_x, food_y 
            poison_coord = poison_x, poison_y

            food.pos = food_coord
            poison.pos = poison_coord


            self.add_widget(food)
            self.add_widget(poison)

            self.food_list.append(food)
            self.poison_list.append(poison)

            self.food_dict[food_coord] = food
            self.poison_dict[poison_coord] = poison

    # going to have to invoke the genetic algorithm here
    def populate_fishes(self):
        fitness_list, total_score = compute_fitness(self.fish_list_genetic_pool)
        selection_pool = compute_selection(fitness_list, total_score)
        self.fish_list_genetic_pool = []
        for i in range(NUM_FISHES):
            new_dna_list = compute_recombination(selection_pool)
            fish = Fish()
            fish.initialize_from_DNA(new_dna_list)
            # print(fish.dna)
            # print(fish.health)
            poison_radius=PoisonRadius()
            food_radius=FoodRadius()
            poison_radius.initialize(fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)
            self.fish_list_genetic_pool.append(fish)

    def populate_food_and_poison(self):
        for child in self.children:
            self.remove_widget(child)
        print(len(self.children))

        self.food_list = []
        self.poison_list =[]
        self.food_dict = {}
        self.poison_dict= {}

        for i in range(int(NUM_FISHES*5)):
            food = Food()
            poison = Poison()

            food_x, food_y = randint(0, 800), randint(0, 600)
            poison_x, poison_y = randint(0, 800), randint(0, 600)

            while poison_x == food_x and poison_y == food_y:
                poison_x, poison_y = randint(0, 800), randint(0, 600)

            food_coord = food_x, food_y 
            poison_coord = poison_x, poison_y

            food.pos = food_coord
            poison.pos = poison_coord


            self.add_widget(food)
            self.add_widget(poison)

            self.food_list.append(food)
            self.poison_list.append(poison)

            self.food_dict[food_coord] = food
            self.poison_dict[poison_coord] = poison

    def serve_fishes(self):
        for fish in self.fish_list:
            fish.center = self.center_x+randint(-200, 200), self.center_y+randint(-200, 200)
            fish.velocity = Vector(randrange(-fish.max_velocity,fish.max_velocity), randrange(-fish.max_velocity,fish.max_velocity))


    def eat_food(self, fish, food_count, food_in_radius, current_pos):
        for coord in self.food_dict:
            if fish.size_val < FOOD_RADIUS:
                if (current_pos[0]<=coord[0]+10 and current_pos[0]>=coord[0]-10) and (current_pos[1]<=coord[1]+10 and current_pos[1]>=coord[1]-10):                     
                    food_coord = randint(0, 800), randint(0, 600)

                    while food_coord in self.food_dict or food_coord in self.poison_dict:
                        food_coord = randint(0, 800), randint(0, 600)

                    new_food = Food()
                    new_food.pos = food_coord
                    self.remove_widget(self.food_dict[coord])
                    self.food_dict.pop(coord, None)
                    self.add_widget(new_food)
                    self.food_dict[food_coord] = new_food

                    fish.health = fish.health + 1
                    fish.fitness += 1
            else:
                if ((coord[0] - current_pos[0])**2 + (coord[1] - current_pos[1])**2) <= (fish.size_val/2)**2:
                    food_coord = randint(0, 800), randint(0, 600)

                    while food_coord in self.food_dict or food_coord in self.poison_dict:
                        food_coord = randint(0, 800), randint(0, 600)


                    new_food = Food()
                    new_food.pos = food_coord
                    self.remove_widget(self.food_dict[coord])
                    self.food_dict.pop(coord, None)
                    self.add_widget(new_food)
                    self.food_dict[food_coord] = new_food

                    fish.health = fish.health + 1
                    fish.fitness += 1

            if coord in self.food_dict:
                distance = compute_distance(current_pos, coord)
                if distance <= fish.food_rad:
                    food_in_radius[distance] = coord
                    food_count += 1

            if len(list(food_in_radius.keys())) > 0:
                min_food_dist = min(list(food_in_radius.keys()))
            else:
                min_food_dist = 0

            if min_food_dist != 0:
                min_food_coord = food_in_radius[min_food_dist]

            if food_count > 0:
                seek = fish.seek(current_pos, min_food_coord)
                seek *= fish.food_propensity
                seek = normalize_magnitude(seek)*fish.max_force
                fish.applyForce(seek)

    def eat_poison(self, fish, poison_count, poison_in_radius, current_pos):
        for coord in self.poison_dict:
            if fish.size_val < POISON_RADIUS:
                if (current_pos[0]<=coord[0]+5 and current_pos[0]>=coord[0]-5) and (current_pos[1]<=coord[1]+5 and current_pos[1]>=coord[1]-5):
                    poison_coord = randint(0, 800), randint(0, 600)

                    while poison_coord in self.food_dict or poison_coord in self.poison_dict:
                        poison_coord = randint(0, 800), randint(0, 600)


                    new_poison = Poison()
                    new_poison.pos = poison_coord
                    self.remove_widget(self.poison_dict[coord])
                    self.poison_dict.pop(coord, None)
                    self.add_widget(new_poison)
                    self.poison_dict[poison_coord] = new_poison

                    fish.health = fish.health*0.5

            # if coord in self.poison_dict:
            else: 
                if ((coord[0] - current_pos[0])**2 + (coord[1] - current_pos[1])**2) <= (fish.size_val/2)**2:
                    poison_coord = randint(0, 800), randint(0, 600)

                    while poison_coord in self.food_dict or poison_coord in self.poison_dict:
                        poison_coord = randint(0, 800), randint(0, 600)

                    new_poison = Poison()
                    new_poison.pos = poison_coord
                    self.remove_widget(self.poison_dict[coord])
                    self.poison_dict.pop(coord, None)
                    self.add_widget(new_poison)
                    self.poison_dict[poison_coord] = new_poison

                    fish.health = fish.health*0.5

            if coord in self.poison_dict:
                distance = compute_distance(current_pos, coord)
                if distance < fish.poison_rad:
                    poison_in_radius[distance] = coord
                    poison_count += 1

            if len(list(poison_in_radius.keys())) > 0:
                min_poison_dist = min(list(poison_in_radius.keys()))
            else:
                min_poison_dist = 0

            if min_poison_dist != 0:
                min_poison_coord = poison_in_radius[min_poison_dist]

            if poison_count > 0:
                seek = fish.seek(current_pos, min_poison_coord)
                seek *= fish.poison_propensity
                seek = normalize_magnitude(seek)*fish.max_force
                fish.applyForce(seek)


    def update(self, dt):
        # print(len(self.poison_dict))
        # print(len(self.food_dict))
        count = 1
        # print("total count: "+ str(self.count))
        
        if self.start_new_gen == True:
            self.gen_count += 1
            self.gen_val = "Generation " + str(self.gen_count)
            self.populate_food_and_poison()
            self.populate_fishes()
            self.serve_fishes()
            self.start_new_gen = False

        else:
            for fish in self.fish_list:
                # print(fish.health)

                # diff = fish.top - fish.pos[1]
                # print("Fish " + str(count)+ " " + "health: " +str(fish.health))
                # print(diff)
                # print(fish.size_val)
                # count += 1

                food_in_radius = {}
                poison_in_radius = {}

                angle_movement = 0

                current_pos = fish.center_x,fish.center_y
                food_count = 0
                poison_count = 0
                # print(current_pos)
                self.eat_food(fish, food_count, food_in_radius, current_pos)
                self.eat_poison(fish, poison_count, poison_in_radius, current_pos)

                # bounce off top and bottom
                if fish.y < 0:
                    fish.y = 0
                    fish.velocity_y *= -1

                elif fish.top > self.height:
                    fish.top = self.height
                    fish.velocity_y *= -1

                if (fish.x < 0):
                    fish.x = 0
                    fish.velocity_x *= -1

                elif (fish.right > self.width):
                    fish.right = self.width
                    fish.velocity_x *= -1


                fish.move()
                for rad in fish.children:
                    rad.move(fish.center_x, fish.center_y)


                if fish.health > 200:
                    fish.health = 200

                fish.health -= 0.2

                if fish.health <= 0:
                    fish.health = 0
                    self.avg_fitness += fish.fitness 
                    for rad in fish.children:
                        fish.remove_widget(rad)

                    self.remove_widget(fish)
                    self.fish_list.remove(fish)

            if len(self.fish_list) == 0:
                self.avg_fitness = self.avg_fitness/len(self.fish_list_genetic_pool)
                print(self.avg_fitness)
                self.avg_fitness = 0
                self.start_new_gen = True




            # print(food_count)
            # print(poison_count)

            # bounce off top and bottom
            


class FishApp(App):
    def build(self):
        game = FishSimulation()
        game.populate_fishes_and_food()
        game.serve_fishes()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game


if __name__ == '__main__':
    FishApp().run()












