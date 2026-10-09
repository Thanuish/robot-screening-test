"""Cloth settings from a TOML file (config/cloth.toml by default).

The file has one table per garment (its size and fabric: every field of cloth.GarmentSpec), plus
the simulation and grasp settings. A garment can inherit another one and change a few values.

    cfg = load()                                   # the file's default garment
    cfg = load("shorts", overrides={"spacing": 0.045})
    cfg = load("jeans", path="my_cloth.toml")

Unknown or misspelled keys are errors (with a suggestion), so a typo never goes unnoticed.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import tomllib
from pathlib import Path

from .cloth import GarmentSpec

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "cloth.toml"


@dataclasses.dataclass
class GPUSettings:
    """[gpu]: the GPU cloth engine (vrteleop/gpu_cloth.py, XPBD)."""
    spacing: float = 0.010             # distance between points [m] (the garment's `spacing` is
                                       # the MuJoCo engine's)
    substeps: int = 20                 # solver substeps per 1/60 s frame
    self_collision_every: int = 2      # self-collision on every n-th substep (the costliest part)
    fps: int = 60
    thickness: float = 0.95            # self-collision distance, as a fraction of spacing
    stretch_compliance: float = 1e-8   # threads (inverse stiffness; 0 = rigid)
    shear_compliance: float = 2e-5     # bias diagonals
    bend_compliance: float = 2e-3      # across every fold line (larger = softer folds)
    cloth_friction: float = 0.5        # Coulomb friction coefficient, fabric on fabric
    table_friction: float = 0.4        # ... fabric on the table
    gripper_friction: float = 0.3      # ... fabric on an arm while its gripper pinches (else 0)
    air_drag: float = 0.5              # [1/s]
    relaxation: float = 1.5            # Jacobi over-relaxation of the constraint solve


@dataclasses.dataclass
class ClothConfig:
    name: str
    garment: GarmentSpec
    gpu: GPUSettings = dataclasses.field(default_factory=GPUSettings)
    # [simulation]
    timestep: float = 0.004
    solver_iterations: int = 30
    # [grasp]
    slip_force: float = 30.0
    release_gap: float = 0.004
    pad_friction: float = 1.0

    def to_json(self) -> str:
        """Everything needed to rebuild this cloth exactly (stored with every episode)."""
        d = dataclasses.asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> "ClothConfig":
        d = json.loads(text)
        g = _coerce_fields(GarmentSpec, d.pop("garment"), "recorded garment")
        gpu = GPUSettings(**_coerce_fields(GPUSettings, d.pop("gpu", {}), "recorded gpu"))
        return cls(garment=GarmentSpec(**g), gpu=gpu, **d)


SECTIONS = {"simulation": ("timestep", "solver_iterations"),
            "grasp": ("slip_force", "release_gap", "pad_friction")}


class ConfigError(ValueError):
    pass


def _suggest(key, valid):
    close = difflib.get_close_matches(key, list(valid), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _coerce_fields(cls, values: dict, where: str) -> dict:
    """Check keys against a dataclass and convert TOML values to the field types."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    out = {}
    for k, v in values.items():
        if k not in fields:
            raise ConfigError(f"{where}: unknown setting '{k}'{_suggest(k, fields)}")
        default = fields[k].default
        if isinstance(default, bool):
            if not isinstance(v, bool):
                raise ConfigError(f"{where}: '{k}' must be true or false, got {v!r}")
        elif isinstance(default, int):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"{where}: '{k}' must be a whole number, got {v!r}")
        elif isinstance(default, float):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{where}: '{k}' must be a number, got {v!r}")
            v = float(v)
        elif isinstance(default, str):
            if not isinstance(v, str):
                raise ConfigError(f"{where}: '{k}' must be text, got {v!r}")
        elif isinstance(default, tuple):
            if not isinstance(v, (list, tuple)) or len(v) != len(default):
                raise ConfigError(f"{where}: '{k}' must be a list of {len(default)} numbers, got {v!r}")
            v = tuple(float(x) for x in v)
        out[k] = v
    return out


def _resolve(garments: dict, name: str, path: Path, chain=()) -> dict:
    """The garment's table merged over the ones it inherits from."""
    if name not in garments:
        raise ConfigError(f"{path}: no garment '{name}'{_suggest(name, garments)}; "
                          f"available: {', '.join(sorted(garments))}")
    if name in chain:
        raise ConfigError(f"{path}: garments inherit in a circle: {' -> '.join(chain + (name,))}")
    table = dict(garments[name])
    parent = table.pop("inherits", None)
    merged = _resolve(garments, parent, path, chain + (name,)) if parent else {}
    for k, v in table.items():
        if (k in SECTIONS or k == "gpu") and isinstance(v, dict):
            merged[k] = {**merged.get(k, {}), **v}
        else:
            merged[k] = v
    return merged


def load(name: str | None = None, path: str | Path | None = None,
         overrides: dict | None = None) -> ClothConfig:
    """Load garment `name` (default: the file's `default`) from `path`; `overrides` replace
    garment fields (e.g. from command-line options)."""
    path = Path(path) if path else DEFAULT_PATH
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"cloth config not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    for k in data:
        if k not in ("default", "garments", "gpu", *SECTIONS):
            raise ConfigError(f"{path}: unknown top-level setting '{k}'"
                              f"{_suggest(k, ['default', 'garments', 'gpu', *SECTIONS])}")
    garments = data.get("garments", {})
    name = name or data.get("default")
    if not name:
        raise ConfigError(f"{path}: no garment given and no 'default = \"...\"' in the file")
    fields = _resolve(garments, name, path)
    settings = {}
    for section, keys in SECTIONS.items():
        values = {**data.get(section, {}), **fields.pop(section, {})}
        for k, v in values.items():
            if k not in keys:
                raise ConfigError(f"{path}: [{section}] unknown setting '{k}'{_suggest(k, keys)}")
            settings[k] = v
    settings = _coerce_fields(ClothConfig, settings, f"{path}")
    gpu = {**data.get("gpu", {}), **fields.pop("gpu", {})}
    gpu = GPUSettings(**_coerce_fields(GPUSettings, gpu, f"{path} [gpu]"))
    g = _coerce_fields(GarmentSpec, fields, f"{path} [garments.{name}]")
    g.update(overrides or {})
    garment = GarmentSpec(**g)
    for k in ("thigh", "hem"):     # wider legs would overlap each other in the flat pattern
        if getattr(garment, k) > garment.hip / 2 + 1e-9:
            raise ConfigError(f"{path} [garments.{name}]: {k} ({getattr(garment, k)}) must be <= "
                              f"hip / 2 ({garment.hip / 2}): the legs would overlap")
    if garment.length - garment.rise < 3 * garment.spacing:
        raise ConfigError(f"{path} [garments.{name}]: the legs (length - rise = "
                          f"{garment.length - garment.rise:.3f} m) need at least 3 x spacing")
    if garment.layer_gap <= 2 * garment.sphere_r:
        raise ConfigError(f"{path} [garments.{name}]: layer_gap ({garment.layer_gap}) must be > "
                          f"2 * sphere_r ({2 * garment.sphere_r})")
    return ClothConfig(name=name, garment=garment, gpu=gpu, **settings)


def names(path: str | Path | None = None) -> list[str]:
    """Garments defined in the file."""
    with open(Path(path) if path else DEFAULT_PATH, "rb") as f:
        return sorted(tomllib.load(f).get("garments", {}))
