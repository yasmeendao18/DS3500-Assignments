"""
DS 3500
Professor Rachlin
Homework 5 Animation
04/14/2023
Yidi Wang, Yasmeen Dao, Lilian Uong
"""
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import copy
import argparse


# Using argparse to get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--grass_rate', default=.025)
parser.add_argument('--fox_k_value', default=10)
parser.add_argument('--field_size', default=300)
parser.add_argument('--initial_rabbits', default=100)
parser.add_argument('--initial_foxes', default=100)
parser.add_argument('--generation', default=1000)
args = parser.parse_args()

class Rabbit:
    """ Rabbit class: a rabbit can reproduce, eat, and move """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        # boolean to determine if rabbit was eaten by fox
        self.got_eaten = False


    def reproduce(self):
        """ Make a new rabbit at the same location.
        Each reproducing rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
            self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-1, 0, 1]))))

class Fox:
    """ Fox class: fox can eat(otherwise starve), reproduce and move. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.days_starved = 0

    def reproduce(self):
        """ Make a new fox at the same location. Each reproducing
         fox's eaten level is reset to zero. """
        self.days_starved = 0
        return copy.deepcopy(self)

    def eat(self):
        """ Feed the fox a rabbit """
        self.days_starved = 0

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-2, 0, 2])) % SIZE
            self.y = (self.y + rnd.choice([-2, 0, 2])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-2, 0, 2]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-2, 0, 2]))))

class Field:
    """ A field is a patch of grass with 0 or more rabbits  """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.rabbits = []
        self.foxes = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.rabbit_loc = np.zeros(shape=(SIZE, SIZE), dtype=int)
        self.nrabbits = []
        self.nfox = []
        self.ngrass = []


    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)
        self.rabbit_loc[rabbit.x, rabbit.y] += 1

    def add_fox(self, fox):
        """ A new fox is added to the field """
        self.foxes.append(fox)

    def move(self):
        """ Animals move """
        for r in self.rabbits:
            r.move()

        for f in self.foxes:
            f.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """

        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x, rabbit.y])
            self.field[rabbit.x, rabbit.y] = 0

        for fox in self.foxes:
            # if the rabbits location is at the fox location
            # fox eats the rabbit
            if self.rabbit_loc[fox.x, fox.y] > 0:
                fox.eat()
                # then the fox moves to the rabbits position
                self.rabbit_loc[fox.x, fox.y] -= 1
            else:
            # fox starves
                fox.days_starved += 1

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """

        # create an array to store the survived rabbits
        survived_rabbits = []
        for r in self.rabbits:
            if r.eaten > 0:
                survived_rabbits.append(r)
            else:
                self.rabbit_loc[r.x, r.y] -= 1
        self.rabbits = survived_rabbits
        
        # fox survive if they starved less than K days
        self.foxes = [f for f in self.foxes if f.days_starved <= K]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:
            for _ in range(rnd.randint(1, OFFSPRING)):
                repr_rabbit = rabbit.reproduce()
                born.append(rabbit.reproduce())
        for r in born:
            self.rabbit_loc[r.x, r.y] += 1
        self.rabbits += born

        # reproduced fox is at most one offspring
        new_fox = []

        for fox in self.foxes:
            for _ in range(rnd.randint(0, 1)):
                if fox.days_starved == 0:
                    new_fox.append(fox.reproduce())
        self.foxes += new_fox

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def num_rabbits(self):
        """ Amount of rabbits in the field """
        return len(self.rabbits)

    def amount_of_grass(self):
        """ Amount of grass in the field """
        return self.field.sum()

    def num_fox(self):
        """ Amount of foxes in the field """
        return len(self.foxes)

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())
        self.nfox.append(self.num_fox())

    def history(self,showPercentage=True):
        """
            Create multi line chart
        :param showPercentage: updates chart to show population percentage
        :return: multi line chart
        """
        
        plt.figure(figsize=(5, 5))
        plt.xlabel("# Generations")
        
        ys_r = self.nrabbits
        ys_g = self.ngrass
        ys_f = self.nfox

        xs_r = [i for i in range(len(self.nrabbits))]
        xs_f = [i for i in range(len(self.ngrass))]
        xs_g = [i for i in range(len(self.nfox))]


        # show the percentage for the y-axis
        if showPercentage:
            maxr = max(ys_r)
            maxg = max(ys_g)
            maxf = max(ys_f)
            ys_r = [y / maxr for y in ys_r]
            ys_g = [y / maxg for y in ys_g]
            ys_f = [y / maxf for y in ys_f]
            plt.ylabel("% Population")

        # plot 3 lines for each species
        plt.plot(xs_r, ys_r, color='blue', label = "rabbit")
        plt.plot(xs_g, ys_g, color='green', label = "grass")
        plt.plot(xs_f, ys_f, color='red', label = "fox")

        # plot the line chart
        plt.grid()
        plt.legend()
        plt.title("Population of rabbit, grass, and fox over generations")
        plt.show()

def animate(i, field, im):
    """
    Create animated chart
    :param i: variable
    :param field: grid where species are on
    :param im: object of field
    :return: animated field
    """
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.field)
    plt.title("generation = " + str(i))
    # store locations of animals
    # np.maximum
    return im

def main():
    # Create the ecosystem
    field = Field()

    for _ in range(INITIAL_RABBITS):
        field.add_rabbit(Rabbit())

    for _ in range(INITIAL_FOXES):
        field.add_fox(Fox())

    # Run the generations
    for _ in range(GENERATION):
        field.generation()

    # call the line chart
    field.history()

    # animate ecosystem with cmap
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    clist = ['white', 'red', 'blue', 'green']
    my_cmap = colors.ListedColormap(clist)
    im = plt.imshow(array, cmap=my_cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im), frames=1000000, interval=1, repeat=True)
    plt.show()

if __name__ == '__main__':
    SIZE = int(args.field_size)  # The dimensions of the field
    OFFSPRING = 2  # Max offspring offspring when a rabbit reproduces
    GRASS_RATE = float(args.grass_rate)  # Probability that grass grows back at any location in the next season.
    WRAP = False
    K = int(args.fox_k_value)
    INITIAL_RABBITS = int(args.initial_rabbits)
    INITIAL_FOXES = int(args.initial_foxes)
    GENERATION = int(args.generation)
    main()
