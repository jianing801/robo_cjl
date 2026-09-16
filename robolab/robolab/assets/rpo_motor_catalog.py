"""Motor choices used by the RPO leg co-design tools.

Actuator torque/speed limits and optional torque-speed envelope files are
applied to Isaac Lab.  Mass, rigid-body inertia, geometry, and URDF data are
deliberately retained as metadata until the parameterized URDF model is
extended separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path


@dataclass(frozen=True)
class MotorSpec:
    """Joint-output specifications for one integrated actuator."""

    motor_id: str
    gear_ratio: float | None
    rated_torque_nm: float | None
    peak_torque_nm: float
    max_speed_rpm: float
    mass_kg: float | None
    envelope_mm: str
    principal_inertia_kgm2: tuple[float, float, float] | None = None
    torque_speed_curve_csv: str | None = None

    @property
    def max_speed_rad_s(self) -> float:
        return self.max_speed_rpm * 2.0 * math.pi / 60.0

    def as_dict(self) -> dict[str, object]:
        return {
            "motor_id": self.motor_id,
            "gear_ratio": self.gear_ratio,
            "rated_torque_nm": self.rated_torque_nm,
            "peak_torque_nm": self.peak_torque_nm,
            "max_speed_rpm": self.max_speed_rpm,
            "max_speed_rad_s": self.max_speed_rad_s,
            "mass_kg": self.mass_kg,
            "envelope_mm": self.envelope_mm,
            "principal_inertia_kgm2": self.principal_inertia_kgm2,
            "torque_speed_curve_csv": self.torque_speed_curve_csv,
        }


# ``legacy`` reproduces the limits that were previously hard-coded in
# roboparty.py.  It is the default for standalone commands so old experiments
# remain reproducible.
MOTOR_CATALOG: dict[str, MotorSpec] = {
    "legacy-knee": MotorSpec(
        "legacy-knee", None, None, 120.0, 25.0 * 60.0 / (2.0 * math.pi),
        None, "URDF unchanged",
    ),
    "legacy-ankle": MotorSpec(
        "legacy-ankle", None, None, 27.0, 8.0 * 60.0 / (2.0 * math.pi),
        None, "URDF unchanged",
    ),
    "RS04": MotorSpec(
        "RS04", 9.0, 35.0, 120.0, 200.0, 1.420, "diameter 120 x 52.2",
        (1.786e-3, 1.862e-3, 2.842e-3),
        "motor_data/RS04_gp_envelope.csv",
    ),
    "DM-J10010L-2EC": MotorSpec(
        "DM-J10010L-2EC", 10.0, 40.0, 120.0, 200.0, 1.372,
        "diameter 120 x 53 (about 55.7 with protrusion)",
        (7.194e-4, 1.103e-3, 1.795e-3),
        "motor_data/DM_J10010L_gp_envelope.csv",
    ),
    "RS06": MotorSpec(
        "RS06", 9.0, 11.0, 36.0, 480.0, 0.621, "diameter 88 x 49",
        (3.591e-4, 3.694e-4, 4.974e-4),
        "motor_data/RS06_gp_envelope.csv",
    ),
    "DM-J4340P-2EC": MotorSpec(
        "DM-J4340P-2EC", 40.0, 9.0, 27.0, 100.0, 0.362,
        "diameter 57 x 53.3",
        (1.444e-4, 1.446e-4, 1.496e-4),
        "motor_data/DM_J4340P_gp_envelope.csv",
    ),
    # Retained for reproducing studies made before the P-version curve was
    # added.  It intentionally keeps the former fixed rectangular limits.
    "DM-J4340-2EC": MotorSpec(
        "DM-J4340-2EC", 40.0, 9.0, 27.0, 100.0, 0.362,
        "diameter 57 x 53.3",
        (1.444e-4, 1.446e-4, 1.496e-4),
    ),
    "DM-J8006-2EC": MotorSpec(
        "DM-J8006-2EC", 6.0, 8.0, 20.0, 200.0, 0.550,
        "diameter 96 x 40",
        (3.680e-4, 3.684e-4, 6.405e-4),
    ),
}

DEFAULT_KNEE_MOTOR = "legacy-knee"
DEFAULT_ANKLE_MOTOR = "legacy-ankle"
KNEE_MOTOR_CANDIDATES = ("RS04", "DM-J10010L-2EC")
ANKLE_MOTOR_CANDIDATES = ("RS06", "DM-J4340P-2EC", "DM-J8006-2EC")
KNEE_MOTOR_CHOICES = (DEFAULT_KNEE_MOTOR, *KNEE_MOTOR_CANDIDATES)
ANKLE_MOTOR_CHOICES = (DEFAULT_ANKLE_MOTOR, *ANKLE_MOTOR_CANDIDATES, "DM-J4340-2EC")


def get_motor_spec(motor_id: str, allowed: tuple[str, ...] | None = None) -> MotorSpec:
    """Return a motor specification, with a useful error for invalid choices."""

    if allowed is not None and motor_id not in allowed:
        raise ValueError(f"Motor '{motor_id}' is not allowed here; choose one of {allowed}.")
    try:
        return MOTOR_CATALOG[motor_id]
    except KeyError as exc:
        raise ValueError(f"Unknown motor '{motor_id}'; choose one of {tuple(MOTOR_CATALOG)}.") from exc


def load_torque_speed_envelope(
    motor: MotorSpec,
    csv_path: str | None = None,
) -> list[list[float]] | None:
    """Load a MATLAB-exported joint-output torque-speed envelope.

    The CSV is consumed directly.  Extra MATLAB diagnostic columns are
    ignored; only ``output_speed_rad_s`` and ``peak_envelope_nm`` are needed.
    """

    selected_path = csv_path or motor.torque_speed_curve_csv
    if selected_path is None:
        return None
    path = Path(selected_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Torque-speed envelope CSV not found: {path}")

    points: dict[float, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"output_speed_rad_s", "peak_envelope_nm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}.")
        for line_number, row in enumerate(reader, start=2):
            try:
                speed = float(row["output_speed_rad_s"])
                torque = float(row["peak_envelope_nm"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid torque-speed value in {path}:{line_number}.") from exc
            if not math.isfinite(speed) or not math.isfinite(torque) or speed < 0.0 or torque < 0.0:
                raise ValueError(f"Non-finite or negative torque-speed value in {path}:{line_number}.")
            points[speed] = min(torque, motor.peak_torque_nm)

    lookup = [[speed, points[speed]] for speed in sorted(points)]
    if len(lookup) < 2 or lookup[0][0] > 1.0e-9:
        raise ValueError(f"{path} must contain at least two points and start at zero speed.")
    speed_tolerance = max(1.0e-6, motor.max_speed_rad_s * 1.0e-5)
    torque_tolerance = max(1.0e-6, motor.peak_torque_nm * 1.0e-6)
    if abs(lookup[-1][0] - motor.max_speed_rad_s) > speed_tolerance:
        raise ValueError(
            f"{path} must end at the catalog max speed "
            f"({motor.max_speed_rad_s:.9g} rad/s)."
        )
    if lookup[-1][1] > torque_tolerance:
        raise ValueError(f"{path} must end at zero torque at the no-load speed.")
    if any(current[1] > previous[1] + torque_tolerance
           for previous, current in zip(lookup, lookup[1:])):
        raise ValueError(f"{path} torque envelope must be monotonically non-increasing.")
    return lookup
