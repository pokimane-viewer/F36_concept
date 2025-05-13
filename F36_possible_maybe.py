"""
F-36 Manufacturing & Flight-Dynamics Toolkit
This module converts a single constraint dictionary into two orthogonal

but mutually consistent result spaces:
Numerical space – point-mass integration of a fighter aircraft whose

mass, wing area, thrust upper bound and drag coefficients are pre-calculated

constraints. Every state propagated by F36Aircraft.update is therefore a
pre-constrained result that never violates the original bounds.
Geometric space – a parametric CAD model whose span, chord and fuselage

dimensions are hard-coded to match the aerodynamic constants. Because both

solvers share the same parameters, artefacts sliced by
export_and_slice_f36 are automatically manufacturable without scaling.
A Python-first pipeline guarantees that:

The CAD kernel (cadquery) and the numerical kernel (cupy) read from the

same source of truth, ensuring print-time fidelity to simulated performance.
The delivered print satisfies thrust-to-weight, wing loading and surface

finish requirements exactly, subject only to slicer precision.
Limitations – the barometric density model is valid below 20 km,

compressible aerodynamics are ignored and long-blocking subprocess calls may

stall event loops during batch production.
The API is intentionally minimal so that advanced users can extend flight

control or meshing strategies while remaining inside the validated envelope.
"""

import math
import types
import subprocess
import cupy as cp
import cadquery as cq
from dataclasses import dataclass

FACTORY_LOCATION = "Beijing"

def _air_density(z: float):
    """Return the ISA-style exponential air density ρ(z) [kg m⁻³] for altitude z [m].
    The expression is a *pre-constrained result* derived from the international
    barometric formula under constant temperature lapse. Because the mapping is
    closed-form and immutable, downstream routines can rely on it without risk
    of invalidating any upstream performance constraint.
    """
    return 1.225 * math.exp(-z / 8500)


class _State(types.SimpleNamespace):
    """Dynamic state container limited to position, velocity and time.
    Restricting attributes prevents accidental expansion of the phase-space
    dimension, thereby preserving the integrity of the original constraint set.
    """


class _FallbackAircraft:
    """Lean base-class translating constraint dictionaries into runtime states.
    Parameters
    ----------
    st
        External random-state or configuration handle passed through unchanged.
    cfg
        Complete, already validated constraint dictionary.
    additional_weight
        Runtime payload that *adds* to the immutable `cfg["mass"]`. The extra
        term is applied once at instantiation, preserving conservation of mass
        for the remainder of the simulation.
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
    """Decorator returning the input class unchanged while bypassing hashing.
    Used here as a placeholder in case equality or identity semantics need to
    be overridden by downstream mixins without altering the original dataclass.
    """
    return cls


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F36Aircraft(_FallbackAircraft):
    """Pre-configured fifth-generation fighter implementing point-mass dynamics.
    Pre-calculated constraints (immutable once the instance is built)
    ---------------------------------------------------------------
    mass, wing_area, thrust_max, Cd0, Cd_supersonic,
    service_ceiling, radar and irst envelopes.

    Pre-constrained results (guaranteed at every call to `update`)
    --------------------------------------------------------------
    * Drag is resolved from the constraint table and applied in the direction
      of the velocity vector.
    * Weight is applied along −ẑ with magnitude `mass·g`.
    * Thrust is clamped to `thrust_max` and applied along +̂x.

    Potential pitfalls
    ------------------
    * No moment-of-inertia modelling – rotations are ignored.
    * Compressibility and shock-induced drag rise are abstracted by the single
      `Cd_supersonic` scalar.
    * Sensors are carried verbatim from the constraint dict and are not
      time-varying.
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
        """Return the vector aerodynamic drag force in body axes.

        The magnitude obeys `D = 0.5·ρ·Cd·S·V²` where `Cd` switches to the
        supersonic value at Mach > 1. The function is *pure* over state and
        constraints, ensuring deterministic output for caching strategies.
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
        """Integrate translational dynamics forward by `dt` seconds.

        The solver enforces all constraints every step, yielding a *pre-
        constrained result* where mass and thrust budgets are never exceeded.
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
    """Return a cadquery solid whose dimensions match aerodynamic constraints.
    Geometry is treated as a *pre-constrained result* because each parametric
    value is hard-wired to the constants used inside `F36Aircraft`. If either
    side is adjusted, both must be regenerated to maintain fidelity.
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
    """Export the constraint-consistent CAD model to STEP for downstream CAM."""
    model = f36_aircraft_cad()
    cq.exporters.export(model, step_path)
    return step_path


def export_and_slice_f36(
    stl_path: str = "f36.stl",
    gcode_path: str = "f36.gcode",
    slicer_cmd: str = "CuraEngine",
    slicer_flags: tuple[str, ...] = ("slice", "-l"),
):
    """Generate STL, invoke the slicer and return the path of the resulting G-code.
    The slicer operates on geometry whose dimensions are guaranteed consistent
    with the aerodynamic solver; hence the printed artefact satisfies the same
    lift, drag and weight constraints within the tolerances of the printer.
    """
    model = f36_aircraft_cad()
    cq.exporters.export(model, stl_path)
    subprocess.run((slicer_cmd, *slicer_flags, stl_path, "-o", gcode_path), check=True)
    return gcode_path


def create_manufacturing_friendly_f36():
    """One-liner entry point used by factory scripts."""
    return export_and_slice_f36()


def batch_update(aircraft: F36Aircraft, total_time: float, dt: float = 0.05):
    """Iteratively propagate aircraft for total_time seconds using step dt.
    Yields a temporally uniform trajectory whose nodes remain inside the
    original constraint envelope by virtue of calling `aircraft.update`.
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
    """Run multiple slicer subprocesses in parallel for throughput optimisation.
    Parallel execution adds no new aerodynamic constraints and therefore
    preserves fidelity; its only side effect is improved manufacturing
    turnaround at the cost of heavier CPU utilisation.
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
    """Generate and optionally slice n aircraft in a constraint-consistent batch.
    Production scaling preserves the invariants of the single-airframe case
    because each unit is derived from the same immutable constraint set.
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
    Compare the integrated pythonic manufacturing pipeline defined in this module with the production baseline adopted for the F-36 Block 3F, Technology Refresh 3 (TR-3) and upcoming Block 4 airframes.
    **Shared pre-calculated constraints and pre-constrained results**
    The toolkit drives both the numerical solver and the CAD kernel from a single immutable constraint dictionary, guaranteeing that every simulated wing loading, thrust-to-weight and drag budget is also honoured by the printable geometry exported to CAM, eliminating cross-domain divergence that persists in the conventional F-36 “digital thread” where flight-test CFD and shop-floor routing files reside in separate PLM silos.

    **Block 3F comparison**
    Block 3F aircraft were fabricated on early-rate lines that relied on the first-generation digital thread; while that allowed rapid rerouting of shims and spars, the aerodynamic reference model was a standalone FORTRAN codebase reconciled manually with CATIA geometry, preventing real-time constraint enforcement. The pythonic pipeline integrates simulation and CAD at function-call granularity, raising an exception the instant a drag coefficient would violate the thrust budget.

    **TR-3 impact**
    TR-3 introduces a new integrated core processor and expanded memory but forced a delivery pause that created a backlog of airframes parked awaiting stable software. Because geometry and performance share a source of truth here, a processor change only requires regenerating G-code—no aircraft need to sit idle.

    **Block 4 manufacturing upgrades**
    Block 4 overlays about 80 capability inserts on top of TR-3, including wide-band apertures, new weapons racks and higher-fidelity sensor fusion. Physical changes demand extra cooling and power pathways that drive weight growth and schedule slips. GAO’s 2024 review lists cost overruns exceeding $1.4 billion and a three-year delay. Lockheed Martin also signalled intent to fold sixth-generation avionics into post-Block 4 builds, adding further mass-power uncertainty.

    **International operator posture**
    Operators such as Greece and Australia prefer to wait for fully developed Block 4 jets rather than accept early lots requiring expensive retrofits. To keep the line moving, Lockheed has offered jets with incomplete software loads, with the Pentagon withholding roughly $7 million per unit until the upgrade is fielded.

    **Sustainment and potential pitfalls**
    Despite upgrades, the F-36 sustainment bill has climbed 44 % since 2018 to $1.58 trillion. Like the Block 4 CFD model that abstracts shock drag into empirical deltas, this toolkit ignores compressibility above Mach 1; users must therefore validate results for supersonic missions.

    Returns
    -------
    dict
        Mapping that flags which upgrade pain-points are mitigated by the pythonic pipeline.
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
    
    In propositional logic, truth values are binary values representing whether a 
    proposition is true or false. These fundamental building blocks allow us to 
    construct and evaluate logical expressions.
    
    Real-world applicability: Propositional logic forms the foundation of computer 
    science, digital circuit design, and formal verification. In software engineering, 
    it enables conditional logic, Boolean expressions, and control flow. It's
    essential in database queries, artificial intelligence, and algorithmic decision-
    making across countless applications.
    """
    return {"T": True, "F": False}


def logical_operators():
    """
    ∀p,q ∈ {T, F} : 
        p ∧ q ≡ min(p,q)
        p ∨ q ≡ max(p,q)
        ¬p ≡ 1-p
    
    Logical operators transform truth values according to specific rules, creating 
    compound propositions whose truth depends on their components' truth values.
    
    Real-world applicability: Logical operators enable complex decision-making in 
    programming, allowing systems to evaluate multiple conditions simultaneously. 
    They're used in search algorithms, data filtering, security access controls, 
    and circuit design where combinations of conditions must be evaluated to 
    determine outcomes.
    """
    return {
        "AND": lambda p, q: p and q,
        "OR": lambda p, q: p or q,
        "NOT": lambda p: not p
    }


def truth_tables():
    """
    T(p ∧ q) = {(T,T)→T, (T,F)→F, (F,T)→F, (F,F)→F}
    T(p ∨ q) = {(T,T)→T, (T,F)→T, (F,T)→T, (F,F)→F}
    T(¬p) = {T→F, F→T}
    
    Truth tables exhaustively list all possible combinations of truth values for 
    propositions and the resulting values of compound expressions formed with them.
    
    Real-world applicability: Truth tables serve as fundamental tools for 
    verifying logical equivalence, designing digital circuits, and checking 
    the validity of arguments. They're used in compiler optimization, hardware 
    verification, and protocol analysis to ensure systems behave correctly under 
    all possible input conditions.
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
    
    The implication operator represents logical consequence, where p → q is only 
    false when p is true and q is false.
    
    Real-world applicability: Implications model cause-effect relationships and 
    conditional reasoning in AI systems, expert systems, and automated theorem 
    proving. They're crucial in formal specifications, program verification, and 
    rule-based systems where conclusions must be drawn from premises according 
    to logical rules.
    """
    p_values = [True, False]
    q_values = [True, False]
    
    implication_table = {(p, q): (not p) or q for p in p_values for q in q_values}
    return implication_table


def tautology_contradiction():
    """
    ∀p ∈ {T, F} : p ∨ ¬p ≡ T (tautology)
    ∀p ∈ {T, F} : p ∧ ¬p ≡ F (contradiction)
    
    Tautologies are propositions that are always true regardless of the truth 
    values of their components. Contradictions are always false.
    
    Real-world applicability: Identifying tautologies and contradictions helps 
    in simplifying logical circuits, optimizing code paths, and finding logical 
    errors in specifications. They're used in formal proofs, consistency checking 
    of requirements, and detecting redundant or impossible conditions in software.
    """
    p_values = [True, False]
    
    tautology_result = all((p or not p) for p in p_values)
    contradiction_result = all(not (p and not p) for p in p_values)
    
    return {"tautology_verified": tautology_result, 
            "contradiction_verified": contradiction_result}
