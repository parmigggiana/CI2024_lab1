from multiprocessing.pool import ThreadPool
import time
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

LIVE_PLOT = False
instances = [
    {  #
        "UNIVERSE_SIZE": 100,
        "NUM_SETS": 10,
        "DENSITY": 0.2,
        "ROUNDS": 1,
        "MUTATION_PERCENT": (0.1, 0.2),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 1,
        "MIN_ITERS": 0,
    },
    {
        "UNIVERSE_SIZE": 1000,
        "NUM_SETS": 100,
        "DENSITY": 0.2,
        "ROUNDS": 3,
        "MUTATION_PERCENT": (0.01, 0.02),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 4,
        "MIN_ITERS": 50,
    },
    {
        "UNIVERSE_SIZE": 10000,
        "NUM_SETS": 1000,
        "DENSITY": 0.2,
        "ROUNDS": 3,
        "MUTATION_PERCENT": (0.001, 0.003),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 5,
        "MIN_ITERS": 60,
    },
    {
        "UNIVERSE_SIZE": 100000,
        "NUM_SETS": 10000,
        "DENSITY": 0.1,
        "ROUNDS": 2,
        "MUTATION_PERCENT": (0.0001, 0.0003),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 6,
        "MIN_ITERS": 70,
    },
    {
        "UNIVERSE_SIZE": 100000,
        "NUM_SETS": 10000,
        "DENSITY": 0.2,
        "ROUNDS": 2,
        "MUTATION_PERCENT": (0.0001, 0.0003),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 6,
        "MIN_ITERS": 70,
    },
    {
        "UNIVERSE_SIZE": 100000,
        "NUM_SETS": 10000,
        "DENSITY": 0.3,
        "ROUNDS": 2,
        "MUTATION_PERCENT": (0.0001, 0.0003),
        "POPULATION_SIZE": 2,
        "WINDOW_SIZE": 6,
        "MIN_ITERS": 70,
    },
]

TOTAL_COST_CALLS = 0


def run(
    UNIVERSE_SIZE,
    NUM_SETS,
    DENSITY,
    ROUNDS,
    MUTATION_PERCENT,
    POPULATION_SIZE,
    WINDOW_SIZE,
    MIN_ITERS,
):
    global TOTAL_COST_CALLS
    HEURISTIC_CHECK_ITERS = int(UNIVERSE_SIZE // (10 ** (np.log10(UNIVERSE_SIZE) // 2)))
    # ic(HEURISTIC_CHECK_ITERS)
    print()
    print(f"UNIVERSE_SIZE: {UNIVERSE_SIZE}")
    print(f"NUM_SETS: {NUM_SETS}")
    print(f"DENSITY: {DENSITY}")

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

        def mutate_single(geneset, min_mutation, max_mutation):
            new_geneset = geneset.copy()
            mutating_genes = rng.integers(
                0,
                NUM_SETS,
                rng.integers(
                    max(NUM_SETS * min_mutation, 1),
                    min(NUM_SETS * max_mutation, NUM_SETS * min_mutation),
                    endpoint=True,
                ),
            )
            new_geneset[mutating_genes] = ~geneset[mutating_genes]
            n = 1
            while n < (3 * HEURISTIC_CHECK_ITERS) and not valid(new_geneset):
                new_geneset[mutating_genes] = geneset[
                    mutating_genes
                ]  # restore original
                mutating_genes = rng.integers(
                    0,
                    NUM_SETS,
                    max(
                        rng.integers(
                            NUM_SETS * min_mutation,
                            NUM_SETS * max_mutation,
                            endpoint=True,
                        ),
                        1,
                    ),
                )
                new_geneset[mutating_genes] = ~geneset[mutating_genes]
                n += 1

            if n == 3 * HEURISTIC_CHECK_ITERS:
                new_geneset = geneset.copy()

            assert valid(new_geneset)
            return new_geneset

        new_genesets = ThreadPool(genesets.shape[0]).map(
            lambda geneset: mutate_single(geneset, min_mutation, max_mutation),
            genesets,
        )
        return np.array(new_genesets)

    print(
        f"Mutating genes: {NUM_SETS * MUTATION_PERCENT[0]:.0f} to {NUM_SETS * MUTATION_PERCENT[1]:.0f}"
    )

    if LIVE_PLOT:
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
        best_geneset = None
        genesets = np.empty((POPULATION_SIZE, NUM_SETS), dtype=bool)
        for i in range(POPULATION_SIZE):
            while not valid(genesets[i]):
                genesets[i] = rng.random(NUM_SETS) < 0.5

        costs = np.array([cost(g) for g in genesets])
        best_cost_idx = np.argmin(costs)
        last_best_cost = max(costs)
        best_costs = [last_best_cost]
        best_geneset = genesets[best_cost_idx]
        iters = 0
        improvement_rate = 1

        if LIVE_PLOT:
            history = [costs]
            # history = [best_costs[-1]]

        while True:
            iters += 1
            population = mutate(
                genesets,
                MUTATION_PERCENT[0],
                MUTATION_PERCENT[1],
            )
            costs = np.array([cost(g) for g in population])
            best_cost_idx = np.argmin(costs)

            if costs[best_cost_idx] < best_costs[-1]:
                best_geneset = population[best_cost_idx]
                last_best_cost = costs[best_cost_idx]
                genesets = np.repeat(
                    population[best_cost_idx][np.newaxis, :], POPULATION_SIZE, axis=0
                )
            best_costs.append(last_best_cost)
            improvement_rate = (
                (best_costs[-(iters // WINDOW_SIZE)] - best_costs[-1])
                / (iters // WINDOW_SIZE)
                / UNIVERSE_SIZE
                if iters > MIN_ITERS
                else 1
            )
            # ic(improvement_rate)

            if LIVE_PLOT:
                history.append(costs)
                # history.append(best_costs[-1])
                if iters % HEURISTIC_CHECK_ITERS == 0:
                    # ax1.cla()
                    ax1.clear()
                    ax1.set_title(
                        f"Round {round}/{ROUNDS} - Best cost: {best_costs[-1]:.2f} - Decrease rate: {improvement_rate:.2f}"
                    )
                    # ax1.set_yscale("log")
                    for i, h in enumerate(history):
                        ax1.scatter(
                            [i] * len(h),
                            h,
                            color="blue",
                            alpha=1 / POPULATION_SIZE,
                            marker=".",
                        )
                    # ax1.plot(history)
                    plt.pause(0.01)

            if improvement_rate <= 0.1:
                # ic(iters)
                break

        if LIVE_PLOT:
            ax1.clear()
            ax1.set_title(
                f"Round {round}/{ROUNDS} - Best cost: {best_costs[-1]:.2f} - Decrease rate: {improvement_rate:.2f}"
            )
            # ax1.set_yscale("log")
            for i, h in enumerate(history):
                ax1.scatter(
                    [i] * len(h),
                    h,
                    color="blue",
                    alpha=1 / POPULATION_SIZE,
                    marker=".",
                )
            # ax1.plot(history)
            plt.pause(0.01)

        total_iterations += iters

        if best_costs[-1] < best_cost_overall:
            best_cost_overall = best_costs[-1]
            best_geneset_overall = best_geneset
            best_round = round
            best_round_iterations = iters
        if best_costs[-1] <= UNIVERSE_SIZE:
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
    print(f"Valid: {valid(best_geneset_overall)}")
    # plt.imshow(SETS[best_geneset_overall])

    plt.show()  # keep it open


if __name__ == "__main__":
    for instance in instances:
        run(**instance)
