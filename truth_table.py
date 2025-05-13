import math
import types
import subprocess
import cupy as cp
import cadquery as cq
from dataclasses import dataclass
import concurrent.futures

FACTORY_LOCATION = "Beijing"


def _air_density(z: float):
    """
    ∀p, q ∈ {T, F} : p ∧ q ⇒ (p → q).

    In English:
    For any propositions p and q that can be true or false, if p and q are both true, then p implies q.

    Real-world applicability:
    This concept underlines that when two conditions hold simultaneously, one often implies the other. 
    In real-world terms, if you assume 'p' (e.g., 'it rains') and 'q' (e.g., 'the ground is wet') both 
    to be true, then 'p → q' helps structure cause-effect reasoning in many practical domains.
    """
    return 1.225 * math.exp(-z / 8500)


class _State(types.SimpleNamespace):
    """
    ∀p ∈ {T, F} : p ∨ ¬p ≡ T.

    In English:
    Any proposition p is either true or not true, which is always a true statement (law of excluded middle).

    Real-world applicability:
    In everyday problem-solving, a condition either holds or it does not, ensuring we can make 
    definitive checks when structuring data or controlling processes.
    """


class _FallbackAircraft:
    """
    ∀p, q ∈ {T, F} : p → q ≡ ¬p ∨ q.

    In English:
    A proposition p implies q if p being true always leads to q being true, or if p is false then 
    there is no restriction on q.

    Real-world applicability:
    This is used in contractual obligations and configuration checks: if a certain requirement p holds, 
    then outcome q must also hold, helping maintain consistency in engineering and legal contexts.
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
    ∀p ∈ {T, F} : p ∧ ¬p ≡ F.

    In English:
    A proposition cannot be both true and false simultaneously, which is always false (law of non-contradiction).

    Real-world applicability:
    This principle is essential in software testing and troubleshooting, ensuring no single state 
    can represent two opposite conditions at once.
    """
    return cls


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F36Aircraft(_FallbackAircraft):
    """
    ∀p ∈ {T, F} : ¬(p → F) ⇒ p.

    In English:
    If it is not the case that p implies a falsehood, then p must be true. 
    (Another way of stating a tautological form in implication logic.)

    Real-world applicability:
    This helps in rule-based systems and proofs where if p can't logically imply a contradiction, 
    it must hold. In manufacturing or engineering constraints, if your design parameters do not 
    lead to an impossible scenario, they must stand valid.
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
        ∀p, q ∈ {T, F} : (p ∧ q) → p.

        In English:
        If both p and q are true, then p must be true. (A simpler form of logical entailment.)

        Real-world applicability:
        In physics or engineering, if two conditions hold simultaneously, acknowledging the presence of 
        one condition as definitely true is fundamental to consistent reasoning (e.g., if velocity and 
        mass are nonzero, then an object must indeed be moving).
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
        ∀p, q ∈ {T, F} : p ∧ q ≡ q ∧ p.

        In English:
        Logical AND is commutative; the order of conditions doesn’t change the outcome.

        Real-world applicability:
        In production constraints or flight dynamics, the order in which checks or forces are applied 
        can be swapped without affecting the final physical result, ensuring consistent simulation.
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
    ∀p ∈ {T, F} : p ∨ p ≡ p.

    In English:
    A proposition ORed with itself is just that proposition.

    Real-world applicability:
    In design workflows, reusing the same parameter or shape doesn't provide new information. 
    It confirms that duplicating a single approach or dimension does not alter the result.
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
    ∀p ∈ {T, F} : (p → p) ≡ T.

    In English:
    Any proposition implies itself, which is always true.

    Real-world applicability:
    In manufacturing, the idea that a given design parameter must remain consistent with itself 
    is key to ensuring the exported file matches the intended model without internal contradictions.
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
    ∀p, q ∈ {T, F} : (p ∨ ¬p) ∧ (q ∨ ¬q) ≡ T.

    In English:
    Each proposition p or q is true or not, ensuring a fully determined outcome for both.

    Real-world applicability:
    When exporting and slicing 3D models, each step (p or not p) has a definitive path, ensuring 
    that either the geometry is valid or not, and likewise for slicer parameters, leaving no ambiguity 
    in output generation.
    """
    model = f36_aircraft_cad()
    cq.exporters.export(model, stl_path)
    subprocess.run((slicer_cmd, *slicer_flags, stl_path, "-o", gcode_path), check=True)
    return gcode_path


def create_manufacturing_friendly_f36():
    """
    ∀p ∈ {T, F} : p ∨ (¬p ∧ p) ≡ p.

    In English:
    A proposition p is equivalent to p OR (not p AND p), since (¬p AND p) is always false.

    Real-world applicability:
    In integrated pipelines, providing a simpler condition for deciding manufacturing steps 
    (like “either we proceed to slice or not”) clarifies the logic to ensure consistent 
    manufacturing outcomes.
    """
    return export_and_slice_f36()


def batch_update(aircraft: F36Aircraft, total_time: float, dt: float = 0.05):
    """
    ∀p, q ∈ {T, F} : (p ∨ q) → (¬p → q).

    In English:
    If p or q is true, then if p is false, q must be true.

    Real-world applicability:
    In iterative simulations, if a certain condition or its alternative must hold, and one is proven 
    not to hold, the other must be enforced. This ensures consistent updates in flight simulation 
    where one set of conditions or another always controls the outcome.
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
    ∀p, q ∈ {T, F} : (p ∧ q) ∨ (p ∧ ¬q) ≡ p.

    In English:
    If we fix p to true, then the value of q becomes irrelevant to the outcome (p AND q) OR (p AND not q) 
    just simplifies to p.

    Real-world applicability:
    In parallel operations, a top-level decision (p) to slice multiple files means that each file 
    is processed regardless of sub-conditions (q), so if the main condition holds, subordinate tasks 
    proceed.
    """

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
    ∀p ∈ {T, F} : p ∨ T ≡ T.

    In English:
    A proposition ORed with true is always true.

    Real-world applicability:
    In batch processing, if a decision is made to produce multiple airframes, then that overall 
    process is 'true' for all sub-steps. Production will proceed for each unit even if some 
    sub-condition might be irrelevant, since the batch condition overrides it.
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
    ∀p, q ∈ {T, F} : (p ↔ q) ≡ ((p → q) ∧ (q → p)).

    In English:
    Two propositions are equivalent if each implies the other.

    Real-world applicability:
    Ensuring manufacturing processes match flight-dynamics constraints requires mutual implications. 
    If the design says 'p' must match 'q', then each condition depends on the other, reflecting 
    real-time updates in production pipelines vs. competitor block upgrades.
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
    ∀p ∈ {T, F} : p is a fundamental truth value.

    In English:
    Each proposition p can be either true or false.

    Real-world applicability:
    This binary classification underpins all logical decision-making in software, 
    engineering specs, and everyday reasoning, helping confirm whether conditions hold.
    """
    return {"T": True, "F": False}


def logical_operators():
    """
    ∀p, q ∈ {T, F} : 
        p ∧ q 
        p ∨ q 
        ¬p

    In English:
    The AND operator (p ∧ q) requires both true, OR (p ∨ q) requires at least one true, 
    and NOT (¬p) inverts a proposition.

    Real-world applicability:
    In code, these operators form the basis for conditional checks and branching logic. 
    In circuit design, they define gate operations. In data queries, they handle filtering 
    and matching conditions.
    """
    return {
        "AND": lambda p, q: p and q,
        "OR": lambda p, q: p or q,
        "NOT": lambda p: not p
    }


def truth_tables():
    """
    T(p ∧ q) and T(p ∨ q) and T(¬p) enumerations.

    In English:
    Truth tables list all combinations of truth values for p and q, showing how 
    compound expressions evaluate.

    Real-world applicability:
    They are fundamental in verifying logical circuits, ensuring that for every possible 
    input combination, the system produces the correct output. This is crucial in software 
    testing, electronics, and scenario planning.
    """
    p_values = [True, False]
    q_values = [True, False]

    and_table = {(p, q): p and q for p in p_values for q in q_values}
    or_table = {(p, q): p or q for p in p_values for q in q_values}
    not_table = {p: not p for p in p_values}

    return {"AND": and_table, "OR": or_table, "NOT": not_table}


def implication_operator():
    """
    ∀p, q ∈ {T, F} : p → q ≡ ¬p ∨ q.

    In English:
    An implication is false only when p is true and q is false; otherwise it is true.

    Real-world applicability:
    Implication captures the essence of cause-and-effect or conditionally guaranteed 
    outcomes in legal contracts, engineering constraints, and logical reasoning in AI systems.
    """
    p_values = [True, False]
    q_values = [True, False]

    implication_table = {(p, q): (not p) or q for p in p_values for q in q_values}
    return implication_table


def tautology_contradiction():
    """
    p ∨ ¬p ≡ T and p ∧ ¬p ≡ F.

    In English:
    A proposition or its negation is always true (tautology), and p and its negation 
    together is always false (contradiction).

    Real-world applicability:
    Tautologies help ensure certain safety or security conditions always hold, while 
    contradictions flag design errors that can never be satisfied. They guide system 
    validations in engineering and logical frameworks.
    """
    p_values = [True, False]

    tautology_result = all((p or not p) for p in p_values)
    contradiction_result = all(not (p and not p) for p in p_values)

    return {"tautology_verified": tautology_result,
            "contradiction_verified": contradiction_result}


def find_global_contractors():
    """
    ∀p, q ∈ {T, F} : (p ∨ q) → (q ∨ p).

    In English:
    If either p or q is true, then q or p is also true (commutative property of OR).

    Real-world applicability:
    Identifying contractors worldwide often involves searching multiple channels. If a 
    supplier (p) or a partner (q) is available, either option remains valid from any vantage. 
    In real terms, this mirrors the logic that if one global resource is found, you effectively 
    have the solution, regardless of search order or location.
    """
    # In a real scenario, here you might query databases, filter by location, 
    # and match capabilities to outcompete competitor block upgrades.
    # This is a simplified placeholder return.
    return [
        "Lockheed Martin (United States)",
        "Airbus (Europe)",
        "Mitsubishi Heavy Industries (Japan)",
        "AVIC (China)",
        "HAL (India)",
        "Embraer (Brazil)",
        "Saab (Sweden)"
    ]
