"""
Engineering Optimization Problems
==================================
5 classic constrained problems using penalty function method.
Each function: obj(x) -> float (including penalty for constraint violations).
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 1. Welded Beam Design (4 variables, 7 constraints)
# ═══════════════════════════════════════════════════════════════

def welded_beam(x: np.ndarray) -> float:
    """Minimize cost of welded beam. x = [h, l, t, b]."""
    h, l, t, b = x[0], x[1], x[2], x[3]
    P = 6000; L = 14; E = 30e6; G = 12e6
    delta_max = 0.25; tau_max = 13600; sigma_max = 30000

    M = P * (L + l / 2)
    R = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
    J = 2 * (np.sqrt(2) * h * l * (l**2 / 4 + ((h + t) / 2)**2))

    tau_prime = P / (np.sqrt(2) * h * l + 1e-30)
    tau_double_prime = M * R / (J + 1e-30)
    tau = np.sqrt(tau_prime**2 + 2 * tau_prime * tau_double_prime * l / (2 * R + 1e-30) + tau_double_prime**2)

    sigma = 6 * P * L / (b * t**2 + 1e-30)
    delta = 6 * P * L**3 / (E * b * t**3 + 1e-30)
    Pc = (4.013 * E * np.sqrt(t**2 * b**6 / 36) / L**2) * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))

    # Objective: cost
    cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

    # Constraints (g <= 0)
    g1 = tau - tau_max
    g2 = sigma - sigma_max
    g3 = h - b
    g4 = 0.10471 * h**2 + 0.04811 * t * b * (14.0 + l) - 5.0
    g5 = 0.125 - h
    g6 = delta - delta_max
    g7 = P - Pc

    penalty = 0
    for g in [g1, g2, g3, g4, g5, g6, g7]:
        if g > 0:
            penalty += 1e6 * g**2
    return cost + penalty

WELDED_BEAM_BOUNDS = (
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([2.0, 10.0, 10.0, 2.0]),
)


# ═══════════════════════════════════════════════════════════════
# 2. Pressure Vessel Design (4 variables, 4 constraints)
# ═══════════════════════════════════════════════════════════════

def pressure_vessel(x: np.ndarray) -> float:
    """Minimize cost of cylindrical pressure vessel. x = [Ts, Th, R, L]."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    cost = (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2
            + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3)

    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
    g4 = x4 - 240

    penalty = 0
    for g in [g1, g2, g3, g4]:
        if g > 0:
            penalty += 1e6 * g**2
    return cost + penalty

PRESSURE_VESSEL_BOUNDS = (
    np.array([0.0625, 0.0625, 10.0, 10.0]),
    np.array([6.1875, 6.1875, 200.0, 200.0]),
)


# ═══════════════════════════════════════════════════════════════
# 3. Tension/Compression Spring Design (3 variables, 4 constraints)
# ═══════════════════════════════════════════════════════════════

def tension_spring(x: np.ndarray) -> float:
    """Minimize weight of tension/compression spring. x = [d, D, N]."""
    d, D, N = x[0], x[1], x[2]

    cost = (N + 2) * D * d**2

    g1 = 1 - D**3 * N / (71785 * d**4 + 1e-30)
    g2 = (4 * D**2 - d * D) / (12566 * (D * d**3 - d**4) + 1e-30) + 1 / (5108 * d**2 + 1e-30) - 1
    g3 = 1 - 140.45 * d / (D**2 * N + 1e-30)
    g4 = (d + D) / 1.5 - 1

    penalty = 0
    for g in [g1, g2, g3, g4]:
        if g > 0:
            penalty += 1e6 * g**2
    return cost + penalty

TENSION_SPRING_BOUNDS = (
    np.array([0.05, 0.25, 2.0]),
    np.array([2.0, 1.3, 15.0]),
)


# ═══════════════════════════════════════════════════════════════
# 4. Speed Reducer Design (7 variables, 11 constraints)
# ═══════════════════════════════════════════════════════════════

def speed_reducer(x: np.ndarray) -> float:
    """Minimize weight of speed reducer. x = [x1..x7]."""
    x1, x2, x3, x4, x5, x6, x7 = x

    cost = (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2) + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2))

    g = np.zeros(11)
    g[0] = 27 / (x1 * x2**2 * x3 + 1e-30) - 1
    g[1] = 397.5 / (x1 * x2**2 * x3**2 + 1e-30) - 1
    g[2] = 1.93 * x4**3 / (x2 * x3 * x6**4 + 1e-30) - 1
    g[3] = 1.93 * x5**3 / (x2 * x3 * x7**4 + 1e-30) - 1
    g[4] = np.sqrt((745*x4/(x2*x3+1e-30))**2 + 16.9e6) / (110*x6**3 + 1e-30) - 1
    g[5] = np.sqrt((745*x5/(x2*x3+1e-30))**2 + 157.5e6) / (85*x7**3 + 1e-30) - 1
    g[6] = x2 * x3 / 40.0 - 1
    g[7] = 5 * x2 / (x1 + 1e-30) - 1
    g[8] = x1 / (12 * x2 + 1e-30) - 1
    g[9] = (1.5 * x6 + 1.9) / (x4 + 1e-30) - 1
    g[10] = (1.1 * x7 + 1.9) / (x5 + 1e-30) - 1

    penalty = sum(1e6 * gi**2 for gi in g if gi > 0)
    return cost + penalty

SPEED_REDUCER_BOUNDS = (
    np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
    np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
)


# ═══════════════════════════════════════════════════════════════
# 5. Three-Bar Truss Design (2 variables, 3 constraints)
# ═══════════════════════════════════════════════════════════════

def three_bar_truss(x: np.ndarray) -> float:
    """Minimize volume of a three-bar truss. x = [A1, A2]."""
    x1, x2 = x[0], x[1]
    l = 100  # length
    P = 2    # load
    sigma = 2  # stress limit

    cost = (2 * np.sqrt(2) * x1 + x2) * l

    g1 = (np.sqrt(2) * x1 + x2) / (np.sqrt(2) * x1**2 + 2 * x1 * x2 + 1e-30) * P - sigma
    g2 = x2 / (np.sqrt(2) * x1**2 + 2 * x1 * x2 + 1e-30) * P - sigma
    g3 = 1 / (np.sqrt(2) * x2 + x1 + 1e-30) * P - sigma

    penalty = sum(1e6 * g**2 for g in [g1, g2, g3] if g > 0)
    return cost + penalty

THREE_BAR_TRUSS_BOUNDS = (
    np.array([0.0, 0.0]),
    np.array([1.0, 1.0]),
)


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════

ENGINEERING_PROBLEMS = {
    "Welded_Beam":      (welded_beam, WELDED_BEAM_BOUNDS, 4),
    "Pressure_Vessel":  (pressure_vessel, PRESSURE_VESSEL_BOUNDS, 4),
    "Tension_Spring":   (tension_spring, TENSION_SPRING_BOUNDS, 3),
    "Speed_Reducer":    (speed_reducer, SPEED_REDUCER_BOUNDS, 7),
    "Three_Bar_Truss":  (three_bar_truss, THREE_BAR_TRUSS_BOUNDS, 2),
}
