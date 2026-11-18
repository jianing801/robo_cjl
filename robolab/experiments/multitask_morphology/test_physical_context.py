"""Check the new equal-width schema against actual inertial coordinates."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from robolab.morphology_context import (
    FIELDS, LINKS, SCHEMA, bind_physical_context, link_values, validate_physical_context,
)


def test_inertial_origin_axes_are_rotated_without_shifting_reference_point():
    root = ET.fromstring(f'''
        <robot><link name="{LINKS[0]}"><inertial>
          <origin xyz="0.1 0.2 0.3" rpy="0 0 {math.pi / 2}"/>
          <mass value="2"/>
          <inertia ixx="2" iyy="3" izz="4" ixy="0.1" ixz="0.2" iyz="0.3"/>
        </inertial></link></robot>''')
    assert link_values(root, LINKS[0]) == pytest.approx([
        2., .1, .2, .3, 3., 2., 4., -.1, -.3, .2,
    ])


def test_context_binds_urdf_values_and_frozen_training_bank_scaling(tmp_path):
    physical = dict(knee_motor="k", ankle_motor="a", selected={
        part: dict(peak_torque_nm=40, max_speed_rad_s=30, armature_kgm2=.02)
        for part in ("knee", "ankle")
    })
    rows = []
    for length, mass in ((.3, 1.), (.4, 2.)):
        urdf = tmp_path/f'{mass}.urdf'
        urdf.write_text('<robot>' + ''.join(f'''
            <link name="{link}"><inertial>
              <origin xyz="0 0 -0.2"/><mass value="{mass}"/>
              <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
            </inertial></link>''' for link in LINKS) + '</robot>')
        rows.append(dict(thigh_m=length, calf_m=.4, urdf=str(urdf), pair_id="k__a",
                         knee_motor="k", ankle_motor="a"))
    bank = dict(designs=rows, motor_pairs={"k__a": physical})
    bind_physical_context(bank)
    assert bank["context_schema"] == SCHEMA and bank["context_width"] == len(FIELDS) == 28
    assert rows[0]["context"][FIELDS.index("thigh_mass_kg")] == -1.
    assert rows[1]["context"][FIELDS.index("calf_mass_kg")] == 1.
    assert rows[1]["context"][FIELDS.index("calf_com_z_m")] == 0.
    validate_physical_context(bank)
    rows[1]["context"][0] = 0.
    with pytest.raises(ValueError, match='does not match'):
        validate_physical_context(bank)


def test_formal_grid_is_exact_3000_subset_without_8006():
    from hybrid_bank import select_grid
    from prepare import select_existing_designs

    here = Path(__file__).resolve().parent
    legacy = (here.parents[2]/'legacy/expert_sources_20261114/robolab/experiments'
              /'rough_hybrid/config.fullbody3000.json')
    old_grid = json.loads(legacy.read_text(encoding='utf8'))['grid']
    target = json.loads((here/'configs/rough.json').read_text(encoding='utf8'))
    rows = select_grid(old_grid, old_grid['knee_pool'], old_grid['ankle_pool'])['designs']
    for row in rows:
        row['pair_id'] = row['knee_motor'] + '__' + row['ankle_motor']
    source = dict(configuration=old_grid, schema_version=2, designs=rows,
                  motor_pairs={row['pair_id']: {} for row in rows})
    retained = select_existing_designs(source, target)
    selected_ids = {row['id'] for row in retained['designs']}
    excluded = [row for row in rows if row['id'] not in selected_ids]
    assert (len(rows), len(selected_ids), len(excluded), len(retained['motor_pairs'])) == (3000,2500,500,25)
    assert {row['ankle_motor'] for row in excluded} == {'DM-J8006-2EC-V1.1'}
