import random

#definisikan nilai awal
strLength = 5
ukuranPopulasi = 10
generasiMaks = 50
kCrossOv = 0.8
mutation_prob = 0.1

def fitness(chromo):
    return chromo.count('1')

def crossover(parent1, parent2):
    cross_point = random.randint(1, strLength - 1)
    child1 = parent1[:cross_point] + parent2[cross_point:]
    child2 = parent2[:cross_point] + parent1[cross_point:]
    return child1, child2

def mutasi(chromosome):
    mutation_point = random.randint(0, strLength - 1)
    mutasid_chromosome = list(chromosome)
    if mutasid_chromosome[mutation_point] == '1':
        mutasid_chromosome[mutation_point] = '0'
    else:
        mutasid_chromosome[mutation_point] = '1'
    return ''.join(mutasid_chromosome)

populasi = []
for i in range(ukuranPopulasi):
    chromosome = ''.join(random.choice(['0', '1']) for _ in range(strLength))
    populasi.append(chromosome)

for generation in range(generasiMaks):
    fitness_scores = []
    for chromosome in populasi:
        fitness_scores.append(fitness(chromosome))

    chromosomeTerb = populasi[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)

    print(f"Generasi {generation}: Chromosome Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}")

    new_populasi = []
    while len(new_populasi) < ukuranPopulasi:
        parent1 = random.choices(populasi, weights=fitness_scores)[0]
        parent2 = random.choices(populasi, weights=fitness_scores)[0]
        if random.random() < kCrossOv:
            child1, child2 = crossover(parent1, parent2)
            new_populasi.append(child1)
            new_populasi.append(child2)
        else:
            new_populasi.append(parent1)
            new_populasi.append(parent2)

    for i in range(ukuranPopulasi):
        if random.random() < mutation_prob:
            new_populasi[i] = mutasi(new_populasi[i])

    populasi = new_populasi

chromosomeTerb = populasi[fitness_scores.index(max(fitness_scores))]
best_fitness = max(fitness_scores)
print(f"\nHasil Akhir: Chromosome Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}")
