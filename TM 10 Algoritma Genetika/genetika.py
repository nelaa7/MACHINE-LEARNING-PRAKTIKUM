import random
import numpy as np  
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Definisikan nilai awal
strLength = 5
ukuranPopulasi = 10
generasiMaks = 50
kCrossOv = 0.8
mutation_prob = 0.1

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Load data from Excel file
# file_path = "pub_england.xlsx"
# file_path = "fertilty.xlsx"
file_path = "geolocation.xlsx"
sheet_name = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Encode the 'Name' column
# encoded_cities = label_encoder.fit_transform(df["name"].astype(str))
# encoded_cities = label_encoder.fit_transform(df["State"].astype(str))
encoded_cities = label_encoder.fit_transform(df["no"].astype(str))


def fitness(chromo):
    return np.sum(
        chromo == 1
    )  # Count the occurrences of 1 (assuming labels are now integers)


def crossover(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    cross_point = random.randint(1, min_length - 1)

    child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]), axis=None)
    child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]), axis=None)

    return child1, child2


def mutasi(chromosome):
    mutation_point = random.randint(0, strLength - 1)
    mutated_chromosome = list(chromosome)
    if mutated_chromosome[mutation_point] == 1:
        mutated_chromosome[mutation_point] = 0
    else:
        mutated_chromosome[mutation_point] = 1
    return mutated_chromosome


# Inisialisasi populasi menggunakan LabelEncoder
populasi = []
for i in range(ukuranPopulasi):
    chromosome = label_encoder.transform(
        # random.sample(df["name"].astype(str).tolist(), strLength)
        # random.sample(df["State"].astype(str).tolist(), strLength)
        random.sample(df["no"].astype(str).tolist(), strLength)
    )
    populasi.append(chromosome)

for generation in range(generasiMaks):
    
    fitness_scores = [fitness(chromosome) for chromosome in populasi]

    # Handle the case where all fitness scores are zero
    if all(score == 0 for score in fitness_scores):
        # Set default weights or take appropriate action
        fitness_scores = [1] * len(fitness_scores)

    chromosomeTerb = populasi[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)

    print(
        # f"Generasi {generation}: Pub_england Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
        # f"Generasi {generation}: Fertilty Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
        f"Generasi {generation}: Location Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
    )

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
print(
    # f"\nHasil Akhir: Pub_england Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
    # f"\nHasil Akhir: Fertilty Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
    f"\nHasil Akhir: Location Terbaik = {chromosomeTerb}, Best fitness = {best_fitness}"
)