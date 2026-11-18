"""
SignalSense FINAL - run with:  python signalsense_final.py

Compares two signal strategies at a 4-road junction (North, South, East, West):

  FIXED    : every road gets the same 30 s green each cycle, no matter what.
  ADAPTIVE : green time is shared in proportion to the queue on each road.
             To make it efficient and fair it also has:
               1. Fairness - a road whose cars have waited long gets extra weight
               2. Gap-out  - green ends early once the queue is empty
               3. Skip     - a road with no cars waiting is skipped

Both strategies get the same maximum green budget (4 x 30 = 120 s per cycle)
and face exactly the same random traffic. Results are averaged over many runs.
Stretch features included: pedestrian phase and peak / off-peak traffic.
"""
import math
import random
from collections import deque

NAMES = ["North", "South", "East", "West"]
RATES = {"North": 0.10, "South": 0.09, "East": 0.04, "West": 0.03}  # cars/sec
FIXED_GREEN = 30
MIN_GREEN = 8         # smallest green given to a road (adaptive)
MAX_GREEN = 60        # largest green given to a road (adaptive)
LOST = 2              # lost start-up seconds at every green
PED_TIME = 10         # pedestrian phase, all cars stopped
FLOW = 0.5            # cars that can leave per green second
HORIZON = 5520        # simulated seconds (about 92 minutes)
RUNS = 20             # random runs to average
FAIR_WEIGHT = 0.15    # extra priority per second the oldest car has waited
SHOW_CYCLES = 6       # cycles printed in detail


def poisson(lam):
    """Random number of cars arriving in one second."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p <= limit:
            return k
        k += 1


def make_traffic(scenario):
    """Arrivals [N, S, E, W] for every second. PEAK = low, high, low demand."""
    traffic = []
    for t in range(HORIZON):
        mult = 1.0
        if scenario == "PEAK":
            mult = 1.4 if HORIZON / 3 <= t < 2 * HORIZON / 3 else 0.6
        traffic.append([poisson(RATES[n] * mult) for n in NAMES])
    return traffic


def get_greens(mode, queues, ages):
    """Planned green seconds for each road in the coming cycle."""
    if mode == "FIXED" or sum(queues) == 0:
        return [FIXED_GREEN] * 4
    budget = FIXED_GREEN * 4
    weights = [q + FAIR_WEIGHT * a if q > 0 else 0 for q, a in zip(queues, ages)]
    spare = budget - MIN_GREEN * 4
    greens = [MIN_GREEN + spare * w / sum(weights) for w in weights]
    return [int(min(g, MAX_GREEN)) if w > 0 else 0 for g, w in zip(greens, weights)]


def simulate(mode, traffic, verbose=False):
    queues = [deque() for _ in range(4)]          # arrival time of each waiting car
    waits = [[] for _ in range(4)]                # wait of each car that got through
    q_total, q_max = [0] * 4, [0] * 4
    t, cycle = 0, 0

    def one_second(active, credit):
        """Advance one second. `active` is the road with green (or None)."""
        nonlocal t
        if t >= HORIZON:
            return credit
        for i in range(4):
            queues[i].extend([t] * traffic[t][i])
        if active is not None:
            credit += FLOW
            while credit >= 1 and queues[active]:
                waits[active].append(t - queues[active].popleft())
                credit -= 1
        for i in range(4):
            q_total[i] += len(queues[i])
            q_max[i] = max(q_max[i], len(queues[i]))
        t += 1
        return credit

    while t < HORIZON:
        start = t
        qn = [len(q) for q in queues]
        ages = [t - q[0] if q else 0 for q in queues]
        greens = get_greens(mode, qn, ages)
        cycle += 1
        if verbose and cycle <= SHOW_CYCLES:
            print(f"  Cycle {cycle:>2} | queues N/S/E/W = {qn} | planned greens = {greens}")
        for i in range(4):
            if mode == "ADAPTIVE" and not queues[i]:
                continue                           # SKIP: nobody waiting
            for _ in range(LOST):
                one_second(None, 0)
            credit = 0.0
            for s in range(greens[i]):
                credit = one_second(i, credit)
                if mode == "ADAPTIVE" and s + 1 >= MIN_GREEN and not queues[i]:
                    break                          # GAP-OUT: queue empty
        for _ in range(PED_TIME):
            one_second(None, 0)
        if t == start:
            one_second(None, 0)

    stats = {}
    for i, name in enumerate(NAMES):
        everyone = waits[i] + [t - a for a in queues[i]]   # include cars still waiting
        stats[name] = {"wait_sum": sum(everyone), "cars": len(everyone),
                       "avg_q": q_total[i] / t, "max_q": q_max[i], "served": len(waits[i])}
    return stats


def run_scenario(scenario):
    """Show one example run in detail, then average many runs."""
    random.seed(0)
    example = make_traffic(scenario)
    for mode in ("FIXED", "ADAPTIVE"):
        print(f"\n {mode} timing (first {SHOW_CYCLES} cycles of one example run):")
        simulate(mode, example, verbose=True)

    total = {m: {n: {"wait_sum": 0, "cars": 0, "avg_q": 0, "max_q": 0, "served": 0}
                 for n in NAMES} for m in ("FIXED", "ADAPTIVE")}
    for seed in range(RUNS):
        random.seed(seed)
        traffic = make_traffic(scenario)           # identical traffic for both modes
        for m in total:
            r = simulate(m, traffic)
            for n in NAMES:
                for k in ("wait_sum", "cars", "avg_q", "served"):
                    total[m][n][k] += r[n][k]
                total[m][n]["max_q"] = max(total[m][n]["max_q"], r[n]["max_q"])
    return total


def report(total):
    print(f"\n  RESULTS (average of {RUNS} runs)")
    print(f"  {'Road':<7}{'Mode':<10}{'AvgQueue':>9}{'MaxQueue':>9}{'AvgWait(s)':>11}{'Cars/run':>9}")
    print("  " + "-" * 55)
    for n in NAMES:
        for m in ("FIXED", "ADAPTIVE"):
            d = total[m][n]
            print(f"  {n:<7}{m.title():<10}{d['avg_q'] / RUNS:>9.2f}{d['max_q']:>9}"
                  f"{d['wait_sum'] / d['cars']:>11.1f}{d['served'] / RUNS:>9.0f}")
    print("\n  Average waiting time per road (each # = 10 seconds)")
    for n in NAMES:
        for m in ("FIXED", "ADAPTIVE"):
            w = total[m][n]["wait_sum"] / total[m][n]["cars"]
            print(f"  {n if m == 'FIXED' else '':<6} {m.title():<9}|{'#' * int(w / 10)} {w:.0f}s")
    res = {}
    for m in total:
        cars = sum(total[m][n]["cars"] for n in NAMES)
        res[m] = (sum(total[m][n]["wait_sum"] for n in NAMES) / cars,
                  sum(total[m][n]["avg_q"] for n in NAMES) / (4 * RUNS),
                  max(total[m][n]["max_q"] for n in NAMES))
    f, a = res["FIXED"], res["ADAPTIVE"]
    print(f"\n  JUNCTION   wait per car : fixed {f[0]:.1f}s -> adaptive {a[0]:.1f}s "
          f"({100 * (f[0] - a[0]) / f[0]:.1f}% better)")
    print(f"             avg queue    : fixed {f[1]:.1f} -> adaptive {a[1]:.1f} "
          f"({100 * (f[1] - a[1]) / f[1]:.1f}% better)")
    print(f"             longest queue: fixed {f[2]} -> adaptive {a[2]}")
    print("  CONCLUSION:", "Adaptive timing HELPS." if a[0] < f[0] else "No benefit.")


def main():
    for scenario in ("NORMAL", "PEAK"):
        print("\n" + "=" * 66)
        print(f" SCENARIO: {scenario} traffic | pedestrian phase ON")
        print("=" * 66)
        report(run_scenario(scenario))
    print()


if __name__ == "__main__":
    main()