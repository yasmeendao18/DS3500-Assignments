import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import copy
import seaborn as sns
import argparse

SIZE = 500  # The dimensions of the field
OFFSPRING = 2  # Max offspring offspring when a rabbit reproduces
GRASS_RATE = 0.025  # Probability that grass grows back at any location in the next season.
WRAP = False  # Does the field wrap around on itself when rabbits move?

# Using argparse to get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--grass_rate', default=0.025)
parser.add_argument('--fox_k_value', default=10)
parser.add_argument('--field_size', default=300)
parser.add_argument('--initial_rabbits', default=100)
parser.add_argument('--initial_foxes', default=100)
parser.add_argument('--generation', default=1000)
args = parser.parse_args()

class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        # boolean to determine if rabbit was eaten by fox
        self.got_eaten = False


    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
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
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.days_starved = 0

    def reproduce(self):
        """ Make a new fox at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the fox a rabbit """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-2, 0, 2])) % SIZE
            self.y = (self.y + rnd.choice([-2, 0, 2])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-2, 0, 2]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-2, 0, 2]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.rabbits = []
        self.all_fox = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.rabbit_init_loc = np.zeros(shape=(SIZE, SIZE), dtype=int)
        self.nrabbits = []
        self.nfox = []
        self.ngrass = []

    # def add_rabbit(self, rabbit):
    #     """ A new rabbit is added to the field """
    #     self.rabbits.append(rabbit)
    #

    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)
        self.rabbit_init_loc[rabbit.x,rabbit.y]+=1

    def add_fox(self, fox):
        """ A new fox is added to the field """
        self.all_fox.append(fox)

    def move(self):
        """ Animals move """
        for r in self.rabbits:
            # new position
            r.move()
            # self.rabbit_init_loc[r.x, r.y] -= 1
            # self.rabbit_init_loc[x_new, y_new] += 1

        for f in self.all_fox:
            f.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """

        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x, rabbit.y])
            self.field[rabbit.x, rabbit.y] = 0

        for fox in self.all_fox:
            # if the rabbits location is at the fox location
            # fox eats the rabbit
            if self.rabbit_init_loc[fox.x, fox.y] > 0:
                fox.eat()
                # then the fox moves to the rabbits position
                self.rabbit_init_loc[fox.x, fox.y] -= 1
            #else:
            # fox starves

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """

        # create an array to store the survived rabbits
        survived_rabbits = []
        for r in self.rabbits:
            if r.eaten > 0 and not r.got_eaten:
                survived_rabbits.append(r)
            else:
                self.rabbit_init_loc[r.x,r.y] -=1
        self.rabbits = survived_rabbits

        self.rabbits = [r for r in self.rabbits if r.eaten > 0]
        self.all_fox = [f for f in self.all_fox if f.eaten > 0]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:
            for _ in range(rnd.randint(1, OFFSPRING)):
                repr_rabbit  = rabbit.reproduce()
                born.append(rabbit.reproduce())
        self.rabbit_init_loc[repr_rabbit.x,repr_rabbit.y] +=1
        self.rabbits += born

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())

        # reproduced fox is at most one offspring
        new_fox = []

        for fox in self.all_fox:
            if fox.days_starved == 0:
                new_fox.append(fox.reproduce())
            self.all_fox += new_fox

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1
        return rabbits

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        #self.nrabbits = len(self.rabbits)
        return self.nrabbits
        # return len(self.rabbits)

    def amount_of_grass(self):
        return self.field.sum()

    def get_fox(self):
        all_fox = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for r in self.all_fox:
            all_fox[r.x, r.y] = 1
        return all_fox

    def num_fox(self):
        """ How many fox are there in the field ? """
        return len(self.all_fox)

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

    def history(self, showTrack=True, showPercentage=True, marker='.'):
        plt.figure(figsize=(10, 6))
        plt.xlabel("# Generations")
        plt.ylabel("# Population")

        # create lists of population values for each species
        rabbit_pop = self.nrabbits[:]
        fox_pop = self.nfoxes[:]
        grass_pop = self.ngrass[:]

        # create x-axis values for each generation
        generations = list(range(len(rabbit_pop)))

        # plot each species' population as a separate line
        plt.plot(generations, rabbit_pop, label='Rabbits')
        plt.plot(generations, fox_pop, label='Foxes')
        plt.plot(generations, grass_pop, label='Grass')

        plt.grid()
        plt.legend()
        plt.title("Simulation Results")

        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    # def history(self, showTrack=True, showPercentage=True, marker='.'):
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.xlabel("# Generations")
    #     plt.ylabel("# Population")
    #
    #     xs = self.nrabbits[:]
    #     ys = int(args.generation)
    #     # ys = self.ngrass[:]
    #
    #     if showTrack:
    #         plt.plot(xs, ys, marker=marker)
    #     else:
    #         plt.scatter(xs, ys, marker=marker)
    #
    #     plt.grid()
    #
    #     plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
    #     plt.savefig("history.png", bbox_inches='tight')
    #     plt.show()

    # def history2(self):
    #     xs = self.nrabbits[:]
    #     ys = self.ngrass[:]
    #
    #     sns.set_style('dark')
    #     f, ax = plt.subplots(figsize=(7, 6))
    #
    #     sns.scatterplot(x=xs, y=ys, s=5, color=".15")
    #     sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
    #     sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
    #     plt.grid()
    #     plt.xlim(0, max(xs) * 1.2)
    #
    #     plt.xlabel("# Rabbits")
    #     plt.ylabel("# Grass")
    #     plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
    #     plt.savefig("history2.png", bbox_inches='tight')
    #     plt.show()


def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.field)
    plt.title("generation = " + str(i))
    # store locations of animals
    #np.maximum
    return im,


def main():
    # Create the ecosystem
    field = Field()

    # # Then God created rabbits....
    # for _ in range(INITIAL_RABBITS):
    #     field.add_rabbit(Rabbit())
    # # And foxes
    # for _ in range(INITIAL_FOXES):
    #     field.add_fox(Fox())

    # update range
    for _ in range(50):
        field.add_rabbit(Rabbit())
    for _ in range(50):
        field.add_fox(Fox())

    # animate ecosystem
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    clist = ['black', 'red', 'blue', 'green']
    my_cmap = colors.ListedColormap(clist)
    im = plt.imshow(array, cmap=my_cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    # field = np.random.randint(0, 2, fig, dtype=int)  # 1 = grass, 0 = bare earth
    # rabbits = np.random.randint(0, 2, fig, dtype=int) * 2  # 2 = rabbit
    # foxes = np.random.randint(0, 2, fig, dtype=int) * 3  # 3 = foxes

    total = np.maximum(field, np.maximum(Rabbit(), Fox()))
    print(field, "\n\n", Rabbit(), "\n\n", Fox(), "\n\n", total)



    plt.imshow(total, cmap=my_cmap, interpolation='none')
    plt.show()

    field.run()
    field.history()


    # field.history2()


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




