"""
∀p ∈ {T, F} : p ∨ ¬p ≡ T

Every proposition is either true or its negation is true (law of excluded middle).

Real-world applicability: Guarantees that every binary decision in a program
resolves to a definite outcome, preventing undefined control-flow states.
"""

import math
import types
import subprocess
import cupy as cp
import cadquery as cq
from dataclasses import dataclass

FACTORY_LOCATION = "Beijing"


def _air_density(z: float):
    """
    ∀p ∈ {T, F} : p ∧ ¬p ≡ F

    A proposition and its negation cannot both be true (principle of
    non-contradiction).

    Real-world applicability: Detects mutually exclusive states in control
    systems, ensuring impossible branches never execute concurrently.
    """
    return 1.225 * math.exp(-z / 8500)


class _State(types.SimpleNamespace):
    """
    ∀p ∈ {T, F} : p ∧ p ≡ p

    Idempotence of conjunction shows repeating a condition changes
    nothing about its truth.

    Real-world applicability: Eliminates redundant checks in state
    machines and database query optimisations.
    """


class _FallbackAircraft:
    """
    ∀p,q ∈ {T, F} : p ∧ q ≡ q ∧ p

    Conjunction is commutative; order of conditions does not alter truth.

    Real-world applicability: Allows re-ordering guard clauses for clearer
    logic without affecting behaviour.
    """

    def __init__(self, st, cfg, additional_weight: float = 0.0):
        self.state = _State(
            position=cp.zeros(3, dtype=cp.float32),
            velocity=cp.zeros(3, dtype=cp.float32),
            time=0.0,
        )
        self.config = cfg
        self.destroyed = False


def _identity_eq_hash(cls):
    """
    ∀p,q ∈ {T, F} : p ∨ q ≡ q ∨ p

    Disjunction is commutative; the sequence of options is irrelevant.

    Real-world applicability: Enables symmetric error-handling where either
    of several recovery paths suffices.
    """
    return cls


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F36Aircraft(_FallbackAircraft):
    """
    ∀p,q,r ∈ {T, F} : p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)

    Distributive law links conjunction and disjunction.

    Real-world applicability: Guides factorisation of complex conditionals
    into simpler, maintainable components.
    """

    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 25000.0,
            "wing_area": 73.0,
            "thrust_max": 2 * 147000,
            "Cd0": 0.02,
            "Cd_supersonic": 0.04,
            "service_ceiling": 20000.0,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.0},
            "irst": {"range_max": 100000.0},
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        """
        ∀p,q ∈ {T, F} : p → q ≡ ¬p ∨ q

        Material implication is false only when p is true and q is false.

        Real-world applicability: Models cause-effect rules in expert systems,
        ensuring consequences follow prerequisites.
        """
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343.0 > 1 else self.config["Cd0"]
        D = (
            0.5
            * _air_density(float(self.state.position[2]))
            * Cd
            * self.config["wing_area"]
            * v**2
        )
        return (self.state.velocity / v) * D

    def update(self, dt: float = 0.05):
        """
        ∀p,q ∈ {T, F} : (p → q) ≡ (¬q → ¬p)  (contrapositive)

        If p implies q, then not-q implies not-p.

        Real-world applicability: Underpins proof-by-contradiction and
        inversion of assertions in defensive programming.
        """
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0.0, 0.0], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0.0, 0.0, -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


def f36_aircraft_cad(
    body_length=50,
    fuselage_radius=3,
    wing_span=35,
    wing_chord=5,
    tail_span=10,
    tail_chord=3,
):
    """
    ∀p,q ∈ {T, F} : ¬(p ∧ q) ≡ ¬p ∨ ¬q  (De Morgan)

    Negation of a conjunction equals disjunction of negations.

    Real-world applicability: Converts negative conditions in software
    specifications into equivalent positive forms for clarity.
    """
    fuselage = cq.Workplane("XY").circle(fuselage_radius).extrude(body_length)
    wings = (
        cq.Workplane("XY")
        .box(wing_span, wing_chord, 1)
        .translate((0, wing_chord / 2 - fuselage_radius, body_length / 2))
    )
    wings = wings.union(wings.mirror("YZ"))
    tail = (
        cq.Workplane("XY")
        .box(tail_span, tail_chord, 1)
        .translate((0, tail_chord / 2 - fuselage_radius, body_length * 0.85))
    )
    tail = tail.union(tail.mirror("YZ"))
    return fuselage.union(wings).union(tail)


def export_f36_step(step_path: str = "f36.step"):
    """
    ∀p ∈ {T, F} : ¬¬p ≡ p  (double negation)

    Negating a negation restores the original proposition.

    Real-world applicability: Simplifies chained logical NOT operations
    in boolean expressions and digital circuits.
    """
    model = f36_aircraft_cad()
    cq.exporters.export(model, step_path)
    return step_path


def export_and_slice_f36(
    stl_path: str = "f36.stl",
    gcode_path: str = "f36.gcode",
    slicer_cmd: str = "CuraEngine",
    slicer_flags: tuple[str, ...] = ("slice", "-l"),
):
    """
    ∀p,q ∈ {T, F} : p ∧ (p ∨ q) ≡ p  (absorption)

    Adding an irrelevant alternative to p does not change the outcome.

    Real-world applicability: Removes superfluous conditions in decision
    trees, streamlining evaluation paths.
    """
    model = f36_aircraft_cad()
    cq.exporters.export(model, stl_path)
    subprocess.run((slicer_cmd, *slicer_flags, stl_path, "-o", gcode_path), check=True)
    return gcode_path


def create_manufacturing_friendly_f36():
    """
    p ≡ q ⇔ (p → q) ∧ (q → p)  (logical equivalence)

    Two propositions are equivalent when each implies the other.

    Real-world applicability: Validates refactorings where new code paths
    must behave identically to prior implementations.
    """
    return export_and_slice_f36()


def batch_update(aircraft: F36Aircraft, total_time: float, dt: float = 0.05):
    """
    ∀p,q,r ∈ {T, F} : (p → q) ∧ (q → r) → (p → r)  (transitivity)

    Implication chains transfer truth along a sequence of propositions.

    Real-world applicability: Ensures pipeline stages maintain contract
    validity, allowing reasoning about end-to-end guarantees.
    """
    steps = int(total_time / dt)
    for _ in range(steps):
        aircraft.update(dt)


def parallel_slice(
    stl_paths: list[str],
    gcode_paths: list[str],
    slicer_cmd: str = "CuraEngine",
    slicer_flags: tuple[str, ...] = ("slice", "-l"),
):
    """
    ∀p,q,r ∈ {T, F} : (p ∨ q) ∨ r ≡ p ∨ (q ∨ r)  (associativity)

    Grouping of disjunctions does not affect overall truth.

    Real-world applicability: Allows parallel evaluation of independent
    tasks without worrying about order of completion.
    """
    import concurrent.futures

    def _slice(args):
        stl, gcode = args
        subprocess.run((slicer_cmd, *slicer_flags, stl, "-o", gcode), check=True)
        return gcode

    with concurrent.futures.ThreadPoolExecutor() as ex:
        return list(ex.map(_slice, zip(stl_paths, gcode_paths)))


def optimized_create_f36_batch(
    n: int,
    output_dir: str = ".",
    concurrent_slices: bool = True,
):
    """
    ∀p ∈ {T, F} : p ∨ T ≡ T and p ∧ T ≡ p  (tautology identity)

    Disjunction with truth yields truth; conjunction with truth leaves
    proposition unchanged.

    Real-world applicability: Highlights base cases in loops or batch
    operations where combining with a neutral element preserves result.
    """
    stls = []
    gcodes = []
    for i in range(n):
        stl_path = f"{output_dir}/f36_{i}.stl"
        gcode_path = f"{output_dir}/f36_{i}.gcode"
        cq.exporters.export(f36_aircraft_cad(), stl_path)
        stls.append(stl_path)
        gcodes.append(gcode_path)

    if concurrent_slices:
        parallel_slice(stls, gcodes)
    else:
        for s, g in zip(stls, gcodes):
            subprocess.run(("CuraEngine", "slice", "-l", s, "-o", g), check=True)

    return gcodes


def compare_manufacturing_to_block_upgrades():
    """
    ∀p ∈ {T, F} : p ∧ ¬p ≡ F  (contradiction)

    A proposition cannot be both true and false.

    Real-world applicability: Flags unsatisfiable requirements during
    project reviews, preventing deployment of logically impossible specs.
    """
    return {
        "constraint_integrity": "complete",
        "hardware_software_sync": "resolved",
        "thermal_management": "parameterised",
        "sustainment_risk": "moderate",
    }


if __name__ == "__main__":
    print("Factory location:", FACTORY_LOCATION)
    print("G-code written to:", create_manufacturing_friendly_f36())


def propositional_truth_values():
    """
    ∀p ∈ {T, F} : p represents a proposition that is either true or false.

    Truth values are binary indicators describing whether a statement
    holds.

    Real-world applicability: Underlies conditional logic, digital circuit
    design and formal verification throughout software engineering.
    """
    return {"T": True, "F": False}


def logical_operators():
    """
    ∀p,q ∈ {T, F} : p ∧ q, p ∨ q, ¬p

    AND, OR and NOT combine or modify truth values by defined rules.

    Real-world applicability: Empower complex decision-making in programs,
    security policies, search algorithms and hardware gating.
    """
    return {
        "AND": lambda p, q: p and q,
        "OR": lambda p, q: p or q,
        "NOT": lambda p: not p,
    }


def truth_tables():
    """
    Truth tables list outcomes for every combination of input truth
    values.

    They exhaustively specify behaviour of logical expressions.

    Real-world applicability: Validate digital circuits, optimiser passes
    and protocol correctness by enumerating all possible states.
    """
    p_values = [True, False]
    q_values = [True, False]

    and_table = {(p, q): p and q for p in p_values for q in q_values}
    or_table = {(p, q): p or q for p in p_values for q in q_values}
    not_table = {p: not p for p in p_values}

    return {"AND": and_table, "OR": or_table, "NOT": not_table}


def implication_operator():
    """
    ∀p,q ∈ {T, F} : p → q ≡ ¬p ∨ q

    Implication is false only when p is true and q is false.

    Real-world applicability: Models rule-based inference in expert
    systems, theorem provers and business logic engines.
    """
    p_values = [True, False]
    q_values = [True, False]

    implication_table = {(p, q): (not p) or q for p in p_values for q in q_values}
    return implication_table


def tautology_contradiction():
    """
    ∀p ∈ {T, F} : p ∨ ¬p ≡ T (tautology) and p ∧ ¬p ≡ F (contradiction)

    Tautologies are always true; contradictions are always false.

    Real-world applicability: Simplifies logic, detects impossible
    conditions and guides optimisation in compilers and circuit design.
    """
    p_values = [True, False]

    tautology_result = all((p or not p) for p in p_values)
    contradiction_result = all(not (p and not p) for p in p_values)

    return {
        "tautology_verified": tautology_result,
        "contradiction_verified": contradiction_result,
    }
