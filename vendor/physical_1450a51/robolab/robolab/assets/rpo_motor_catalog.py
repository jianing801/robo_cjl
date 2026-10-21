"""Motor choices used by the RPO leg co-design tools.

Actuator torque/speed limits, joint-side armature, optional torque-speed
envelopes, and the matching motor-dependent URDF fit are selected together.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path


_DM_SELECTION_PATH = Path(__file__).resolve().parent / "motor_data/selected/DM_selection_2026_09_23.csv"
_DM_SELECTION_NUMERIC_FIELDS = frozenset({
    "source_page", "rated_output_torque_nm", "peak_output_torque_nm",
    "reported_no_load_speed_rpm", "no_load_output_speed_48v_rpm",
    "reported_rated_speed_rpm", "gear_ratio", "mass_kg",
    "rated_phase_current_48v_a", "peak_phase_current_48v_a", "rated_power_w",
    "pole_pairs", "slots", "line_to_line_resistance_ohm",
    "line_to_line_inductance_uh", "flux_pm_wb",
    "reported_rotational_inertia_kgm2", "gear_coefficient",
    "reported_torque_constant_nm_per_a", "driver_max_current_a",
    "protocol_pmax_rad", "protocol_vmax_rad_s", "protocol_tmax_nm",
    "reported_speed_constant",
})


def _load_dm_selection_parameters() -> dict[str, dict[str, object]]:
    """Read the supplied factory sheet while preserving its parameter labels.

    Line-to-line R/L is not silently converted to phase/dq parameters.
    'Rotational inertia' has no axis definition in this selection sheet.
    """
    with _DM_SELECTION_PATH.open(newline="", encoding="utf-8-sig") as stream:
        records = {}
        for row in csv.DictReader(stream):
            motor_id = row["motor_id"]
            if motor_id in records:
                raise ValueError(f"Duplicate factory selection motor: {motor_id}")
            records[motor_id] = {
                name: (float(value) if value else None)
                if name in _DM_SELECTION_NUMERIC_FIELDS else value
                for name, value in row.items()
            }
        return records


_DM_SELECTION_PARAMETERS = _load_dm_selection_parameters()


def _load_dm_parameter_models() -> dict[str, dict[str, str]]:
    """Load generated-model provenance separately from raw factory definitions."""
    path = Path(__file__).resolve().parent / "motor_data/selected/DM_parameter_models_2026_10_09.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = {}
        for row in csv.DictReader(stream):
            if row["motor_id"] in records:
                raise ValueError(f"Duplicate Damiao parameter model: {row['motor_id']}")
            records[row["motor_id"]] = row
        return records


_DM_PARAMETER_MODELS = _load_dm_parameter_models()


@dataclass(frozen=True)
class MotorSpec:
    """Joint-output specifications for one integrated actuator."""

    motor_id: str
    gear_ratio: float | None
    rated_torque_nm: float | None
    peak_torque_nm: float
    max_speed_rpm: float | None
    mass_kg: float | None
    envelope_mm: str
    principal_inertia_kgm2: tuple[float, float, float] | None = None
    torque_speed_curve_csv: str | None = None
    armature_kgm2: float | None = 0.01
    armature_source: str = "legacy fixed value"
    urdf_model_id: str | None = None
    source_url: str | None = None
    data_notes: str = ""
    rotor_inertia_kgm2: float | None = None
    model_kind: str = "existing_model"
    model_label: str = ""

    @property
    def display_name(self) -> str:
        """Human-readable version label; never an alias for the motor ID."""
        return self.model_label or self.motor_id

    def __post_init__(self) -> None:
        if self.armature_kgm2 is not None and (
            not math.isfinite(self.armature_kgm2) or self.armature_kgm2 <= 0.0
        ):
            raise ValueError(
                f"Motor '{self.motor_id}' armature_kgm2 must be finite and positive."
            )

    @property
    def max_speed_rad_s(self) -> float | None:
        if self.max_speed_rpm is None:
            return None
        return self.max_speed_rpm * 2.0 * math.pi / 60.0

    def missing_actuator_inputs(self) -> tuple[str, ...]:
        """Inputs needed before the actuator can be tested independently."""
        missing = []
        if self.max_speed_rpm is None:
            missing.append("48 V no-load output speed")
        if self.torque_speed_curve_csv is None:
            missing.append("48 V peak torque-speed CSV")
        if self.armature_kgm2 is None:
            missing.append("joint-side armature or explicitly accepted inertia model")
        return tuple(missing)

    def missing_co_design_inputs(self) -> tuple[str, ...]:
        """Inputs needed for the complete motor + SW-URDF BO path."""
        missing = list(self.missing_actuator_inputs())
        if self.urdf_model_id is None:
            missing.append("motor-specific thigh/calf SW-URDF GP model")
        return tuple(missing)

    def as_dict(self) -> dict[str, object]:
        return {
            "motor_id": self.motor_id,
            "display_name": self.display_name,
            "gear_ratio": self.gear_ratio,
            "rated_torque_nm": self.rated_torque_nm,
            "peak_torque_nm": self.peak_torque_nm,
            "max_speed_rpm": self.max_speed_rpm,
            "max_speed_rad_s": self.max_speed_rad_s,
            "mass_kg": self.mass_kg,
            "envelope_mm": self.envelope_mm,
            "principal_inertia_kgm2": self.principal_inertia_kgm2,
            "torque_speed_curve_csv": self.torque_speed_curve_csv,
            "armature_kgm2": self.armature_kgm2,
            "armature_source": self.armature_source,
            "urdf_model_id": self.urdf_model_id,
            "source_url": self.source_url,
            "data_notes": self.data_notes,
            "rotor_inertia_kgm2": self.rotor_inertia_kgm2,
            "model_kind": self.model_kind,
            "parameter_model": self.parameter_model,
            "manufacturer_parameters": self.manufacturer_parameters,
        }


    @property
    def manufacturer_parameters(self) -> dict[str, object] | None:
        """A copy of source values; source definitions remain distinct from models."""
        source = _DM_SELECTION_PARAMETERS.get(self.motor_id)
        return None if source is None else dict(source)

    @property
    def parameter_model(self) -> dict[str, str] | None:
        """Model assumptions and generated-curve checks, not measured truth."""
        model = _DM_PARAMETER_MODELS.get(self.motor_id)
        return None if model is None else dict(model)


def _staged_motor(
    motor_id: str,
    gear_ratio: float | None,
    rated_torque_nm: float | None,
    peak_torque_nm: float,
    max_speed_rpm: float | None,
    mass_kg: float | None,
    source_url: str,
    *,
    rotor_inertia_gcm2: float | None = None,
    envelope_mm: str = "verify manufacturer STEP and robot installation",
    notes: str = "",
    urdf_model_id: str | None = None,
) -> MotorSpec:
    """Catalog a manufacturer specification without inventing a BO-ready model.

    Manufacturer rotor inertia is *not* automatically a complete Isaac Lab
    armature: the gearbox/controller-side definition must first be confirmed.
    """
    selection = _DM_SELECTION_PARAMETERS.get(motor_id)
    if selection is not None:
        gear_ratio = selection["gear_ratio"]
        rated_torque_nm = selection["rated_output_torque_nm"]
        peak_torque_nm = selection["peak_output_torque_nm"]
        max_speed_rpm = selection["no_load_output_speed_48v_rpm"]
        mass_kg = selection["mass_kg"]
        envelope_mm = str(selection["dimensions_mm"])
        notes = (
            "Factory selection sheet 2026-09-23 (page 1): 48 V supply supported; "
            "source magnetic flux, line-to-line R/L, phase current and rotational "
            "inertia values are available in manufacturer_parameters. "
            "Phase/dq conversion and inertia axis remain unconfirmed. "
            "Protocol VMAX/TMAX and driver maximum current are not physical output limits. "
            "A 48 V peak envelope and motor-specific assembled SW fit are still required."
        )
        if max_speed_rpm is None:
            notes += " Listed 260 rpm has no voltage label; 48 V speed remains unconfirmed."
        if motor_id == "DM-J8009P-2EC-V1.1":
            notes += " Public 2024 STEP still needs V1.1 matching."
    model = _DM_PARAMETER_MODELS.get(motor_id)
    modeled = model is not None and model["model_status"] == "parameter_model"
    if modeled:
        # A model endpoint is distinct from a manufacturer-confirmed 48 V speed.
        # This permits an explicitly labelled, unanchored electrical model while
        # preserving the unknown manufacturer value in manufacturer_parameters.
        max_speed_rpm = float(model["no_load_48v_rpm"])
        notes = notes.replace(
            "A 48 V peak envelope and motor-specific assembled SW fit are still required.",
            ("Motor-specific assembled SW-GP model is available."
             if urdf_model_id is not None else
             "Motor-specific assembled SW-GP model is still required."),
        )
        notes += (
            " Generated 48 V MCB/SPMSM + constrained-GP parameter envelope is available. "
            "User accepted the family Rs/Ls convention, peak-current convention and "
            f"motor-side Inertia assumption; model record dated {model['model_date']}. Not a measured full-speed curve; "
            "armature is conditional J*N^2 without independently verified gearbox contribution. "
            +
            ("A segment-matched assembled-SW GP model is available for full co-design."
             if urdf_model_id is not None else
             "A motor-specific assembled-SW GP model is still required for full co-design.")
        )
        if model.get("speed_endpoint_source") == "mcb_natural_48v_no_anchor":
            notes += (
                " Runtime speed endpoint is computed by the unanchored 48 V MCB model, "
                "not a manufacturer-verified no-load speed; the unspecified-voltage "
                "260 rpm is NOT used as a 48 V anchor."
            )
        elif model.get("speed_endpoint_source") == "user_assumed_48v_anchor":
            notes = notes.replace(
                " Listed 260 rpm has no voltage label; 48 V speed remains unconfirmed.",
                " Factory 260 rpm has no voltage label; user explicitly assigns it to 48 V for this model.",
            )
            notes += (
                " Runtime endpoint is 260 rpm by user-assumed 48 V anchor, not a manufacturer-verified voltage attribution. "
                "Effective rated phase current is 22 A from manual V1.3 dated 2026-10-15, page 6; "
                "the historical 2026-09-23 factory transcription remains unchanged. "
                "Peak phase current stays 70 A and output peak cap stays 45 Nm."
            )
    return MotorSpec(
        motor_id=motor_id,
        gear_ratio=gear_ratio,
        rated_torque_nm=rated_torque_nm,
        peak_torque_nm=peak_torque_nm,
        max_speed_rpm=max_speed_rpm,
        mass_kg=mass_kg,
        envelope_mm=envelope_mm,
        torque_speed_curve_csv=model["curve_csv"] if modeled else None,
        armature_kgm2=float(model["armature_kgm2"]) if modeled else None,
        armature_source=(
            "accepted parameter-model assumption: source Inertia motor-side * gear_ratio^2; gearbox contribution unverified"
            if modeled else "pending joint-side inertia validation"
        ),
        source_url=source_url,
        data_notes=notes,
        rotor_inertia_kgm2=(
            float(model["assumed_motor_side_inertia_kgm2"]) if modeled else
            (None if rotor_inertia_gcm2 is None else rotor_inertia_gcm2 * 1e-7)
        ),
        model_kind="parameter_model" if modeled else "pending_model",
        urdf_model_id=urdf_model_id,
        model_label=(
            f"{selection['source_model_label']} ({peak_torque_nm:g} N·m)"
            if selection is not None else motor_id
        ),
    )


# Keep RoboParty's original rectangular defaults for its unchanged workflow.
# Added versioned IDs are independent motors, not aliases for these defaults.
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
        "motor_data/torque_speed_curves/RS04_gp_envelope.csv",
        armature_kgm2=0.04000,
        armature_source="manufacturer low-speed equivalent inertia",
        urdf_model_id="RS04_thigh",
    ),
    "DM-J10010L-2EC": MotorSpec(
        "DM-J10010L-2EC", 10.0, 40.0, 120.0, 200.0, 1.375,
        "diameter 120 x 53 (about 55.7 with protrusion)",
        (7.194e-4, 1.103e-3, 1.795e-3),
        "motor_data/torque_speed_curves/DM_J10010L_gp_envelope.csv",
        armature_kgm2=0.05556,
        armature_source="0.0005556 kg*m^2 motor inertia reflected by 10^2",
        urdf_model_id="DM10010L_thigh",
        rotor_inertia_kgm2=0.0005556,
        data_notes="Factory 2026-09-23 mass is about 1375 g. Existing calibration inertia 0.0005556 and fitted URDF are retained; sheet rounds inertia to 0.000556.",
        model_label="DM-J10010L-2EC L版（48 V，120 N·m；非J10010）",
    ),
    "RS06": MotorSpec(
        "RS06", 9.0, 11.0, 36.0, 480.0, 0.621, "diameter 88 x 49",
        (3.591e-4, 3.694e-4, 4.974e-4),
        "motor_data/torque_speed_curves/RS06_gp_envelope.csv",
        armature_kgm2=0.01200,
        armature_source="manufacturer low-speed equivalent inertia",
        urdf_model_id="RS06_calf",
    ),
    "DM-J4340P-2EC": MotorSpec(
        "DM-J4340P-2EC", 40.0, 9.0, 27.0, 100.0, 0.381,
        "diameter 57 x 56.5",
        (1.444e-4, 1.446e-4, 1.496e-4),
        "motor_data/torque_speed_curves/DM_J4340P_gp_envelope.csv",
        armature_kgm2=0.03200,
        armature_source="0.00002 kg*m^2 motor inertia reflected by 40^2",
        urdf_model_id="DM4340P_calf",
        rotor_inertia_kgm2=0.00002,
        data_notes="Factory 2026-09-23 corrects P-version reference mass/outline to 381 g and 57 x 56.5 mm. Existing assembled SW fit, curve and legacy principal-inertia estimates are unchanged.",
        model_label="DM-J4340P-2EC 原款（非V1.1，48 V，27 N·m）",
    ),
    # Additional DM models listed in the manufacturer's current product center.
    # A product listing verifies the catalog torque, not the 48 V peak curve,
    # 52 V battery compatibility, a joint-side inertia, or current stock.
    "DM-J10010-2EC": _staged_motor(
        "DM-J10010-2EC", 10.0, 40.0, 150.0, 150.0, 1.485,
        "https://www.damiaokeji.com/index.php?c=show&id=107",
        envelope_mm="maximum outline 120 x 62; verify installation",
        notes="Manual: 48 V, 21 pole pairs, 0.13 ohm, 100 uH phase inductance, CAN 1 Mbps. Distinct from J10010L (120 Nm); official 48 V / 100 rpm chart is a fixed-speed test, not a full peak envelope. Suggested overvoltage protection <=52 V leaves no battery/regeneration margin.",
        urdf_model_id="DM10010_thigh",
    ),
    "DM-J6248P-2EC": _staged_motor(
        "DM-J6248P-2EC", 48.0, 30.0, 97.0, None, 0.628,
        "https://www.damiaokeji.com/index.php?c=show&id=110",
        envelope_mm="diameter 76 x 62.5; verify installation",
        notes="Manual: 24-48 V, 14 pole pairs, 0.17 ohm, 85 uH phase inductance, 60 rpm with no voltage label; CAN up to 5 Mbps. Confirm classic CAN 1 Mbps support and 52 V bus suitability.",
        urdf_model_id="DM6248P_thigh",
    ),
    "DM-J10422P-2EC": _staged_motor(
        "DM-J10422P-2EC", 22.0, 100.0, 400.0, 120.0, 2.700,
        "https://www.damiaokeji.com/index.php?c=show&id=126",
        envelope_mm="diameter 110 x 89.9; verify installation",
        notes="Manual: 48 V, 16 pole pairs, 0.081 ohm, 38 uH phase inductance, CAN up to 5 Mbps. Very heavy; classic CAN 1 Mbps and installation must be verified.",
        urdf_model_id="DM10422P_thigh",
    ),
    "DM-J8009P-2EC-V1.1": _staged_motor(
        "DM-J8009P-2EC-V1.1", 9.0, 20.0, 40.0, 335.0, 0.950,
        "https://www.damiaokeji.com/index.php?c=show&id=137",
        envelope_mm="diameter 98 x 61.7; verify installation",
        notes="V1.1 manual: 335 rpm at 48 V, 21 pole pairs, 0.145 ohm, 80 uH phase inductance, CAN 1 Mbps. Published 2024 STEP must not be assumed to represent V1.1. Suggested overvoltage protection <=52 V has no bus/regeneration margin.",
        urdf_model_id="DM8009P_V11_calf",
    ),
    "DM-J4340P-2EC-V1.1-48V": _staged_motor(
        "DM-J4340P-2EC-V1.1-48V", 40.0, 12.0, 40.0, 112.0, 0.381,
        "https://www.damiaokeji.com/index.php?c=show&id=134",
        notes="New V1.1, not the existing 9/27 Nm model. Manual's TN chart is 24 V at fixed 36 rpm, not a 48 V peak envelope. Confirm 52 V bus/regeneration protection.",
        urdf_model_id="DM4340P_V11_48V_calf",
    ),
    "DM-J8006-2EC-V1.1": _staged_motor(
        "DM-J8006-2EC-V1.1", 6.0, 8.0, 20.0, None, 0.559,
        "https://www.damiaokeji.com/index.php?c=show&id=103",
        notes="V1.1 manual repeats @24V on both no-load-speed rows; 48 V speed is unresolved. Suggested overvoltage protection <=52 V conflicts with a 48-52 V bus without margin.",
        urdf_model_id="DM8006_V11_calf",
    ),
    "DM-J4310P-2EC-48V": _staged_motor(
        "DM-J4310P-2EC-48V", 10.0, 3.5, 12.5, 450.0, 0.330,
        "https://www.damiaokeji.com/index.php?c=show&id=109",
        envelope_mm="diameter 57 x 49.05; verify installation",
        notes="48 V manual column: 450 rpm, 14 pole pairs, about 0.58 ohm and 320 uH phase inductance, CAN 1 Mbps. Do not mix with 24 V 200 rpm variant; verify 52 V bus and regeneration protection.",
        urdf_model_id="DM4310P_48V_calf",
    ),
    "DM-J6216P-2EC": _staged_motor(
        "DM-J6216P-2EC", 16.0, 10.0, 45.0, None, 0.740,
        "https://github.com/dmBots/DM-J6216P-2EC",
        notes="Added by user request for ankle selection. Use the latest supplied sheet's 45 Nm, not the older manual's 48 Nm. Manufacturer 48 V no-load speed is unconfirmed; use the user-approved 260 rpm 48 V model endpoint. Its matching calf SW-GP fit is available.",
        urdf_model_id="DM6216P_calf",
    ),
}

# Reference-only models must not re-enter any training/evaluation/BO entry.
NOT_SELECTED_MOTOR_REASONS = {
    "DM-J8520P-2EC": "Not selected; unresolved 176 Nm factory peak versus 129.99168 Nm parameter-model plateau; archived by user request.",
    "AK10-9-V2": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AK10-9-V3": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AK60-39-V3": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AK70-9-V3": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AK80-64": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AKH70-16": "Not selected; incomplete cross-brand parameters are archived separately.",
    "AKH70-48": "Not selected; incomplete cross-brand parameters are archived separately.",
    "DM-J4310-2EC-V1.2-48V": "P version selected instead: DM-J4310P-2EC-48V",
    "DM-J4340-2EC-V1.1-48V": "P version selected instead: DM-J4340P-2EC-V1.1-48V",
    "DM-J8009-2EC": "P version selected instead: DM-J8009P-2EC-V1.1",
    "RMD-X6-P20-60": "Not selected; incomplete cross-brand parameters are archived separately.",
    "RMD-X8-P20-120-E": "Not selected; incomplete cross-brand parameters are archived separately.",
    "Unitree-B1-16": "Not selected; incomplete cross-brand parameters are archived separately.",
    "Unitree-IM6014": "Not selected; incomplete cross-brand parameters are archived separately.",
}

DEFAULT_KNEE_MOTOR = "legacy-knee"
DEFAULT_ANKLE_MOTOR = "legacy-ankle"
KNEE_MOTOR_CANDIDATES = ("RS04", "DM-J10010L-2EC")
ANKLE_MOTOR_CANDIDATES = ("RS06", "DM-J4340P-2EC")
# These are *design-variable IDs*, not claims that the added models are ready
# to train.  co_design.py preflights every selected ID before creating a study.
DM_STAGED_KNEE_IDS = (
    "DM-J10010-2EC", "DM-J6248P-2EC",
    "DM-J10422P-2EC",
)
DM_STAGED_ANKLE_IDS = (
    "DM-J8009P-2EC-V1.1", "DM-J4340P-2EC-V1.1-48V",
    "DM-J8006-2EC-V1.1", "DM-J4310P-2EC-48V",
    "DM-J6216P-2EC",
)
KNEE_MOTOR_DESIGN_CANDIDATES = (
    *KNEE_MOTOR_CANDIDATES,
    *DM_STAGED_KNEE_IDS,
)
ANKLE_MOTOR_DESIGN_CANDIDATES = (
    *ANKLE_MOTOR_CANDIDATES,
    *DM_STAGED_ANKLE_IDS,
)
# Explicit BO pools: only segment-matched models with a motor envelope, armature,
# and a per-motor assembled-SW GP fit. Incomplete entries remain visible in the
# design catalog but are rejected by co_design.py before creating a study.
KNEE_MOTOR_BO_CANDIDATES = KNEE_MOTOR_DESIGN_CANDIDATES
ANKLE_MOTOR_BO_CANDIDATES = ANKLE_MOTOR_DESIGN_CANDIDATES
STAGED_MOTOR_IDS = frozenset(
    (*KNEE_MOTOR_DESIGN_CANDIDATES, *ANKLE_MOTOR_DESIGN_CANDIDATES)
) - frozenset((*KNEE_MOTOR_CANDIDATES, *ANKLE_MOTOR_CANDIDATES))
KNEE_MOTOR_CHOICES = (DEFAULT_KNEE_MOTOR, *KNEE_MOTOR_DESIGN_CANDIDATES)
ANKLE_MOTOR_CHOICES = (DEFAULT_ANKLE_MOTOR, *ANKLE_MOTOR_DESIGN_CANDIDATES)


def get_motor_spec(motor_id: str, allowed: tuple[str, ...] | None = None) -> MotorSpec:
    """Return a motor specification, with a useful error for invalid choices."""

    if motor_id in NOT_SELECTED_MOTOR_REASONS:
        raise ValueError(
            f"Motor '{motor_id}' is not selected: {NOT_SELECTED_MOTOR_REASONS[motor_id]}"
        )
    if allowed is not None and motor_id not in allowed:
        raise ValueError(f"Motor '{motor_id}' is not allowed here; choose one of {allowed}.")
    try:
        return MOTOR_CATALOG[motor_id]
    except KeyError as exc:
        raise ValueError(f"Unknown motor '{motor_id}'; choose one of {tuple(MOTOR_CATALOG)}.") from exc


def _raise_if_missing(motor: MotorSpec, missing: tuple[str, ...], target: str) -> None:
    if missing:
        raise ValueError(
            f"Motor '{motor.motor_id}' is staged, not {target}-ready; missing: "
            + "; ".join(missing)
            + ". Do not replace these with a rectangular limit or another motor's URDF."
        )


def require_motor_actuator_ready(motor: MotorSpec) -> None:
    """Fail before Isaac Sim startup if the motor's actuator model is incomplete."""
    _raise_if_missing(motor, motor.missing_actuator_inputs(), "actuator")


def require_motor_ready(motor: MotorSpec) -> None:
    """Fail before BO startup if the actuator or assembled SW fit is incomplete."""
    _raise_if_missing(motor, motor.missing_co_design_inputs(), "BO")


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
    if motor.max_speed_rad_s is None:
        raise ValueError(f"Motor '{motor.motor_id}' needs a 48 V no-load speed before curve loading.")
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
