from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty, StringProperty, ObjectProperty, NumericProperty, ReferenceListProperty, DictProperty, BooleanProperty
from kivy.lang import Builder
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics import *
from random import randint
from random import randrange
from copy import deepcopy
import numpy as np
import math
from math import acos
from math import sqrt
from math import pi



#loads in the Kivy file
Builder.load_file('fish.kv')




# Global Constants for the Radii of the food and the poison
POISON_RADIUS = 10
FOOD_RADIUS = 10

# Global Constant for the number of fishes
NUM_FISHES = 10

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
    health = NumericProperty(200)
    colorLowBound = NumericProperty(200-int(200/13))


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

    # TODO: DNA once the variables above have been initialized
    dna = StringProperty('')

    # Function to initialize the poison perception radius, food perception radius and the size of the fish
    # TODO: Need to create an initialize that rolls from the DNA
    def initialize(self):
        self.poison_rad = randint(1,128)
        self.food_rad = randint(1,128)
        self.size_val = randint(1, 64)
        self.food_propensity = 0.5
        self.poison_propensity = -0.2

    def seek(self, current_pos,target_pos):
        desired_velocity = Vector(target_pos) - Vector(current_pos)
        desired_velocity = normalize_magnitude(desired_velocity)*self.max_velocity
        steering_force = desired_velocity - Vector(self.velocity)
        steering_force = normalize_magnitude(steering_force)*self.max_force
        return(steering_force)

    def applyForce(self, force):
        print(force)
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
            self.colorLowBound -= 200/13

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

    fish_list = ListProperty([])

    food_list = ListProperty([])
    poison_list = ListProperty([])

    food_dict = DictProperty({})
    poison_dict = DictProperty({})

    gen_val = StringProperty("Generation 1")
    gen_count = NumericProperty(1)

    count = NumericProperty(0)

    start_new_gen = BooleanProperty(False)

    def populate_balls_and_food(self):

        ''' Debugging to print the height and the width of the  actual screen'''
        # print(self.height)
        # print(self.width)
        for i in range(NUM_FISHES):
            fish = Fish()
            fish.initialize()
            poison_radius=PoisonRadius()
            food_radius=FoodRadius()
            poison_radius.initialize(fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)
            print(fish.color_val)
        for i in range(int(NUM_FISHES*10)):
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

    def populate_balls(self):
        for i in range(NUM_FISHES):
            fish = Fish()
            fish.initialize()
            poison_radius=PoisonRadius()
            food_radius=FoodRadius()
            poison_radius.initialize(fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)

    def serve_balls(self):
        for fish in self.fish_list:
            fish.center = self.center
            fish.velocity = Vector(randrange(-fish.max_velocity,fish.max_velocity), randrange(-fish.max_velocity,fish.max_velocity))

            # new = deepcopy(fish.velocity)


    def eat_food(self, fish, food_count, food_in_radius, current_pos):
        for coord in self.food_dict:
            if fish.size_val < FOOD_RADIUS:
                if (current_pos[0]<=coord[0]+5 and current_pos[0]>=coord[0]-5) and (current_pos[1]<=coord[1]+5 and current_pos[1]>=coord[1]-5):                     
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
            else:
                if coord[0] <= current_pos[0]+fish.size_val/2 and coord[0]>=current_pos[0]-fish.size_val/2:
                    if coord[1] <= current_pos[1]+fish.size_val/2 and coord[1]>=current_pos[1]-fish.size_val/2:
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

                    fish.health = fish.health - 5

            # if coord in self.poison_dict:
            else: 
                if coord[0] <= current_pos[0]+fish.size_val/2 and coord[0]>=current_pos[0]-fish.size_val/2:
                    if coord[1] <= current_pos[1]+fish.size_val/2 and coord[1]>=current_pos[1]-fish.size_val/2:
                        poison_coord = randint(0, 800), randint(0, 600)

                        while poison_coord in self.food_dict or poison_coord in self.poison_dict:
                            poison_coord = randint(0, 800), randint(0, 600)

                        new_poison = Poison()
                        new_poison.pos = poison_coord
                        self.remove_widget(self.poison_dict[coord])
                        self.poison_dict.pop(coord, None)
                        self.add_widget(new_poison)
                        self.poison_dict[poison_coord] = new_poison

                        fish.health = fish.health - 5

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
            self.populate_balls()
            self.serve_balls()
            self.start_new_gen = False

        else:
            for fish in self.fish_list:

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
                    for rad in fish.children:
                        fish.remove_widget(rad)
                    self.remove_widget(fish)
                    self.fish_list.remove(fish)

            if len(self.fish_list) == 0:
                self.start_new_gen = True




            # print(food_count)
            # print(poison_count)

            # bounce off top and bottom
            


class FishApp(App):
    def build(self):
        game = FishSimulation()
        game.populate_balls_and_food()
        game.serve_balls()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game


if __name__ == '__main__':
    FishApp().run()












