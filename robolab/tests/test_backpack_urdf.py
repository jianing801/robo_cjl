"""Pure-XML checks for the rigid backpack's mass and inertia bookkeeping."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[1] / "robolab" / "tasks" / "direct" / "backpack_walk" / "backpack_urdf.py"
spec = spec_from_file_location("backpack_urdf_under_test", SOURCE)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_backpack_is_separate_mass_randomizable_link_and_source_is_unchanged(tmp_path):
    original = tmp_path / "source.urdf"
    loaded = tmp_path / "loaded.urdf"
    original.write_text(
        '<robot name="test"><link name="torso_link"><inertial>'
        '<origin xyz="0 0 0" rpy="0 0 0"/><mass value="2"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        '</inertial></link></robot>', encoding="utf-8",
    )
    before = original.read_bytes()
    result = module.write_backpack_urdf(
        original, loaded, 2.0, size_m=(0.2, 0.2, 0.2), center_m=(-0.2, 0, 0),
    )
    assert original.read_bytes() == before
    assert result["backpack_mass_body"] == "backpack_link"
    assert result["urdf_import_must_preserve_fixed_link"] is True
    root = ET.parse(loaded).getroot()
    torso = root.find("link[@name='torso_link']")
    pack = root.find("link[@name='backpack_link']")
    assert torso.find("inertial/mass").get("value") == "2"
    assert torso.find("inertial/origin").get("xyz") == "0 0 0"
    assert len(torso.findall("visual")) == 0
    assert pack.find("inertial/mass").get("value") == "2"
    assert pack.find("inertial/origin").get("xyz") == "0 0 0"
    assert len(pack.findall("visual")) == 1
    assert len(pack.findall("collision")) == 1
    inertia = pack.find("inertial/inertia")
    expected_axis = 2.0 * (0.2**2 + 0.2**2) / 12
    assert math.isclose(float(inertia.get("ixx")), expected_axis)
    assert math.isclose(float(inertia.get("iyy")), expected_axis)
    assert math.isclose(float(inertia.get("izz")), expected_axis)
    mount = root.find("joint[@name='backpack_mount_joint']")
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == "torso_link"
    assert mount.find("child").get("link") == "backpack_link"
    assert mount.find("origin").get("xyz") == "-0.2 0 0"


def test_cannot_overwrite_source(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text("<robot/>", encoding="utf-8")
    try:
        module.write_backpack_urdf(source, source, 1.0)
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("Source overwrite was not rejected")


def test_each_environment_gets_reproducible_mass_properties_with_fixed_collision_box():
    first = module.sample_backpack_properties(32, seed=17, mass_range_kg=(5.0, 15.0))
    second = module.sample_backpack_properties(32, seed=17, mass_range_kg=(5.0, 15.0))
    assert first == second
    assert [geometry["env_id"] for geometry in first] == list(range(32))
    assert len({geometry["inertia_equivalent_size_m"] for geometry in first}) == 32
    for geometry in first:
        assert geometry["center_m"] == module.DEFAULT_CENTER_M
        assert geometry["size_m"] == module.DEFAULT_SIZE_M
        assert 5.0 <= geometry["mass_kg"] <= 15.0
        assert all(0.9 * base <= value <= 1.1 * base for value, base in zip(
            geometry["inertia_equivalent_size_m"], module.DEFAULT_SIZE_M,
        ))


def test_geometry_sampler_rejects_invalid_ranges():
    for kwargs in (
        {"count": 0},
        {"count": 2, "inertia_size_scale_range": (0.0, 1.0)},
    ):
        try:
            module.sample_backpack_properties(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid geometry settings to fail: {kwargs}")
