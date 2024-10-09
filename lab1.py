import time

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

UNIVERSE_SIZE = 1000
NUM_SETS = 100
DENSITY = 0.2

ROUNDS = 10
MUTATION_PERCENT = (
    1 / 10 ** (np.log10(UNIVERSE_SIZE) / 2),
    3 / 10 ** (np.log10(UNIVERSE_SIZE) / 2),
)
POPULATION_SIZE = 3
PLOT = False


rng = np.random.Generator(
    np.random.PCG64([UNIVERSE_SIZE, NUM_SETS, int(10_000 * DENSITY)])
)

# DON'T EDIT THESE LINES!

SETS = np.random.random((NUM_SETS, UNIVERSE_SIZE)) < DENSITY
for s in range(UNIVERSE_SIZE):
    if not np.any(SETS[:, s]):
        SETS[np.random.randint(NUM_SETS), s] = True
COSTS = pow(SETS.sum(axis=1), 1.1)

TOTAL_COST_CALLS = 0


def valid(solution):
    return np.all(np.logical_or.reduce(SETS[solution]))


def cost(solution):
    global TOTAL_COST_CALLS
    TOTAL_COST_CALLS += 1
    return COSTS[solution].sum()


def mutate(
    genesets,
    min_mutation,
    max_mutation,
):
    """Tweaks the solution by adding/removing a random set"""
    new_genesets = genesets.copy()
    for new_geneset in new_genesets:
        mutating_genes = None
        while mutating_genes is None or not valid(new_geneset):
            mutating_genes = rng.integers(
                0,
                NUM_SETS,
                max(
                    rng.integers(
                        UNIVERSE_SIZE * min_mutation,
                        UNIVERSE_SIZE * max_mutation,
                        endpoint=True,
                    ),
                    1,
                ),
            )
            # ic(mutating_genes)
            new_geneset[mutating_genes] = ~new_geneset[mutating_genes]

    return new_genesets


print(
    f"Mutating genes: {UNIVERSE_SIZE * MUTATION_PERCENT[0]:.0f} to {UNIVERSE_SIZE * MUTATION_PERCENT[1]:.0f}"
)


if PLOT:
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    # ax1.scatter([])

best_geneset_overall = None
best_cost_overall = float("inf")
best_round = 0
best_round_iterations = 0
total_iterations = 0

start = time.time()
for round in range(1, ROUNDS + 1):
    # Init
    print(f"Round {round}/{ROUNDS}")
    genesets = np.empty((POPULATION_SIZE, NUM_SETS), dtype=bool)
    for i in range(POPULATION_SIZE):
        geneset = None
        while not valid(geneset):
            geneset = rng.random(NUM_SETS) < 0.5
        genesets[i] = geneset
    costs = np.array([cost(g) for g in genesets])
    best_cost_idx = np.argmin(costs)
    last_best_cost = max(costs)
    best_diff = last_best_cost - costs[best_cost_idx]
    iters = 0
    stale_iterations = 0
    temperature = 1.0
    decrease_rate = 1
    if PLOT:
        history = [costs]
        # history = [last_best_cost]

    while True:
        iters += 1
        stale_iterations += 1
        decrease_rate = best_diff / stale_iterations
        population = mutate(
            genesets,
            MUTATION_PERCENT[0] * temperature,
            MUTATION_PERCENT[1] * temperature,
        )
        # genesets, costs = select(population, n=POPULATION_SIZE)
        costs = np.array([cost(g) for g in population])
        best_cost_idx = np.argmin(costs)

        if costs[best_cost_idx] < last_best_cost:
            best_diff = (last_best_cost - costs[best_cost_idx] + best_diff) / 2
            stale_iterations = 0
            last_best_cost = costs[best_cost_idx]
            genesets = np.repeat(
                population[best_cost_idx][np.newaxis, :], POPULATION_SIZE, axis=0
            )

        if iters % (UNIVERSE_SIZE / 2 / (10 ** (np.log10(UNIVERSE_SIZE) / 2))) == 0:
            if decrease_rate < 50:
                temperature /= 1.1
            else:
                temperature *= 1.1
        if PLOT:
            history.append(costs)
            # history.append(last_best_cost)

        if (
            PLOT
            and iters % (UNIVERSE_SIZE / 2 / (10 ** (np.log10(UNIVERSE_SIZE) / 2))) == 0
        ):
            # ax1.cla()
            ax1.clear()
            ax1.set_title(
                f"Round {round}/{ROUNDS} - Best cost: {last_best_cost:.2f} - Temperature: {temperature:.2f}"
            )
            ax1.set_yscale("log")
            for i, h in enumerate(history):
                ax1.scatter(
                    [i] * len(h), h, color="blue", alpha=1 / POPULATION_SIZE, marker="."
                )
            # ax1.plot(history)
            plt.pause(0.01)

        if decrease_rate < 3:
            break
    total_iterations += iters

    if last_best_cost < best_cost_overall:
        best_cost_overall = last_best_cost
        best_geneset_overall = genesets[0]
        best_round = round
        best_round_iterations = iters
    if last_best_cost <= UNIVERSE_SIZE:
        print("Found optimal solution")
        break
elapsed = time.time() - start

minutes = int(elapsed // 60)
seconds = elapsed % 60
print(f"Best cost: {best_cost_overall}")
print(f"Total cost calls: {TOTAL_COST_CALLS}")
print(f"Elapsed time: {minutes}m {seconds:.2f}s")
print(f"Total number of iterations: {total_iterations}")
print(f"Best round: {best_round}/{ROUNDS}")
print(f"Iterations in round {best_round}: {best_round_iterations}")

print(f"Selected sets: {list(np.nonzero(best_geneset_overall))}")
# plt.imshow(SETS[best_geneset_overall])

plt.show()  # keep it open
