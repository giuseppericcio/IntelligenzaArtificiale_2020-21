# -*- coding: utf-8 -*-
import numpy as np
import genetic_algorithm as ga
#import math

""" function, x dev'essere compreso tra -100 e 100 """
def F(x) :
    if (x <= 5.2):
        return 10
    if ((x >= 5.2) and (x <= 20)):
        return x * x
    if (x > 20):
        #return math.cos(x) + 160 * x  tralasciamo il coseno che è ininfluente o quasi
        return 160 * x

#print(F(100))

""" Inputs of the function. """
function_inputs = np.random.randint(-500,500,100)
#print("function_inputs: \n", function_inputs)

"""Number of the weights we are looking to optimize."""
num_weights = len(function_inputs)
#print(num_weights)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 8
num_parents_mating = 4

""" Defining the population size."""
pop_size = (sol_per_pop,num_weights)
"""Creating the initial population."""
new_population = np.random.uniform(low=-500, high=500, size=pop_size)
#print("POPULATION \n" , new_population)

""" Performing GA"""
best_outputs = []
num_generations = 100
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(function_inputs, new_population)
    print("Fitness")
    print(fitness)

    best_outputs.append(np.max(np.sum(new_population * function_inputs, axis=1)))
    # The best result in the current iteration.
    print("Best result : ", np.max(np.sum(new_population * function_inputs, axis=1)))

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness,
                                    num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0] - parents.shape[0], num_weights))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(function_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

import matplotlib.pyplot

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
