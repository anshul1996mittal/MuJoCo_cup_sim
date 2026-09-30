"""All tunable parameters of the cup-lifting study, in SI units.

Every number that matters for the physics lives here so that it can be
defended (and changed) in one place.  Command-line flags in run_trial.py /
run_all.py override a subset of these.
"""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class CupParams:
    # Geometry of the mid-surface of the cup wall (typical ~200 ml disposable cup,
    # matches the photo in the brief). The rim is ignored, as allowed.
    r_bottom: float = 0.025        # bottom radius  [m]  (50 mm base)
    r_top: float = 0.036           # top radius     [m]  (72 mm mouth)
    height: float = 0.085          # height         [m]
    thickness: float = 0.0005      # wall thickness [m]  (brief: <= 1 mm; disposable cups 0.2-0.5 mm)

    # Material: polypropylene (PP) - typical disposable-cup plastic
    young: float = 1.5e9           # Young's modulus [Pa]  (PP: 1.3 - 1.8 GPa)
    poisson: float = 0.42          # Poisson ratio        (PP: ~0.40 - 0.45)
    density: float = 905.0         # [kg/m^3] -> mass is computed from area * t * rho
    damping: float = 2e-4          # Rayleigh (stiffness-proportional) damping [s]
    rest_shape_correction: bool = True   # cancel MuJoCo's flat-rest bending bias (see sim.py)
    calibrate_bending: bool = True       # match plate-theory bending rigidity (see calibration.py)

    # Discretisation (triangle shell mesh)
    n_theta: int = 24              # vertices around the circumference
    n_wall: int = 8                # segments along the height
    n_bottom: int = 3              # radial rings on the bottom disk

    # Contact
    mu_finger: float = 0.5         # friction finger-pad <-> cup (plastic on plastic, dry)
    solref: tuple = (0.002, 1.0)   # contact time-constant / damping ratio (stiff)
    solimp: tuple = (0.99, 0.999, 0.0005, 0.5, 2.0)


@dataclass
class CubeParams:
    size: float = 0.030            # edge length [m] (same for every trial -> only mass changes)
    mass: float = 0.100            # [kg]  overridden per trial: 0.010 / 0.100 / 0.500
    mu_cup: float = 0.4            # friction cube <-> cup (plastic on plastic)


@dataclass
class TaskParams:
    cup_xy: tuple = (0.55, 0.0)    # cup position on the floor [m] (robot base at origin)
    grasp_height: float = 0.055    # height of the TCP (pad centre) above the floor [m]
                                   # (pads straddle the wall ~30 mm below the mouth)
    lift_height: float = 0.10      # vertical lift distance [m]
    approach_height: float = 0.08  # pre-grasp: TCP this far above the grasp point [m]

    # Grasp force policy
    # "auto": F = max(f_min, safety * W / (2 * mu))   (W = weight of cup + cube)
    # a number: that constant squeeze force per finger [N] for every trial
    grip_force: object = "auto"
    grip_safety: float = 1.5
    grip_force_min: float = 2.0

    # Timeline [s]
    t_settle: float = 0.5          # cup & cube settle on the floor
    t_approach: float = 1.5        # vertical descent over the cup
    t_preclose: float = 0.6        # fingers move (position mode) to just touch the wall
    t_squeeze: float = 1.5         # grip force ramps 0 -> F  (objective 1)
    t_hold_ground: float = 1.0     # hold squeeze on the ground (deformation reaches steady state)
    t_lift: float = 2.5            # vertical lift (minimum-jerk profile) (objective 2/3)
    t_hold_air: float = 2.0        # static hold in the air
    # Slip test (objective 4): ramp grip force F -> -3 N (fingers open) over t_release,
    # the trial stops shortly after gross slip (cup drops > 2 mm w.r.t. the pads)
    slip_test: bool = True
    t_release: float = 4.0


@dataclass
class SimParams:
    timestep: float = 1e-3
    integrator: str = "discrete"   # MuJoCo 3.14: implicit flex elasticity (M + h^2 K)
    solver: str = "Newton"
    cone: str = "elliptic"         # exact Coulomb cone (not the pyramidal approximation)
    impratio: float = 100.0        # friction rows 100x "harder" than normal rows -> less creep
    log_hz: float = 200.0          # data logging rate
    snapshot_fps: int = 30         # full-state snapshots saved for replay.py


@dataclass
class Config:
    cup: CupParams = field(default_factory=CupParams)
    cube: CubeParams = field(default_factory=CubeParams)
    task: TaskParams = field(default_factory=TaskParams)
    sim: SimParams = field(default_factory=SimParams)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(dct):
        cfg = Config()
        for section, values in dct.items():
            sub = getattr(cfg, section)
            for k, v in values.items():
                if not hasattr(sub, k):          # ignore keys from older versions
                    continue
                if isinstance(getattr(sub, k), tuple):
                    v = tuple(v)
                setattr(sub, k, v)
        return cfg

    @staticmethod
    def from_json(path):
        with open(path) as f:
            return Config.from_dict(json.load(f))


TRIAL_MASSES = {"10g": 0.010, "100g": 0.100, "500g": 0.500}
