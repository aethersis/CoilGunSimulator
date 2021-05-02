import os
import shutil
import string
from copy import copy
from math import pi
import random
from pathlib import Path
from typing import List

import femm
from dataclasses import dataclass
import numpy as np

from src.coil_inductance import calc_multi_layer_for_known_layer_count
from src.spice_model import CircuitModel, CircuitParameters
from src.utils import generate_unique_file_name, cleanup


@dataclass
class SimulationParameters:
    coil_inner_diameter_mm: float = 10.0
    projectile_diameter_mm: float = 8.5
    wire_without_insulation_thickness_mm: float = 1.0
    wire_with_insulation_thickness_mm: float = 1.07
    capacitance_uF: float = 518.0
    capacitor_ESR_ohm: float = 0.09
    capacitor_voltage_V: float = 380
    coil_length_mm: float = 30
    coil_layers_count: int = 5
    projectile_density_g_mm3 = 0.00768
    projectile_length_mm: float = 25
    projectile_initial_position_mm: float = 1
    initial_velocity_m_s: float = 0.0


@dataclass
class Constraint:
    range_from: float
    range_to: float
    current_value: float
    step_size: float


@dataclass
class Constraints:
    coil_length_mm: Constraint = Constraint(20, 40, 30, 1)
    coil_layers_count: Constraint = Constraint(1, 8, 5, 1)
    projectile_length_mm: Constraint = Constraint(18, 42, 25, 1)
    initial_projectile_position_mm: Constraint = Constraint(0, 6, 1, 1)


@dataclass
class SimulationResult:
    end_velocity_m_s: float
    end_kinetic_energy_J: float
    velocity_plot: List[float]
    energy_plot: List[float]
    timescale: List[float]


class FEMMModel:
    @staticmethod
    def _init_femm(femm_file: str) -> Path:
        if not os.path.exists("tmp") and not os.path.isdir("tmp"):
            os.mkdir("tmp")

        femm.openfemm()
        femm.opendocument(femm_file)

        # This part is necessary to run the simulation (possibly in separate threads), otherwise the base file would be overwritten
        unique_file_path = generate_unique_file_name("tmp", ".fem")
        femm.mi_saveas(unique_file_path.as_posix())
        femm.mi_seteditmode("group")
        femm.mi_analyze()

        return unique_file_path

    def __init__(
        self,
        constraints: SimulationParameters,
        circuit_model: CircuitModel,
        femm_file: str = "models/femm/coilgun.fem",
        start_minimized: bool = True,
        dt: float = 0.00005,
        t_max: float = 0.004,
    ):
        """
        Loads FEMM model from file and runs the simulation for the predefined constraints, outputting the output kinetic
        energy of the projectile and its velocity.
        :param constraints: constraints of the simulation such as coil shape, projectile shape etc.
        :param femm_file: femm file that contains the predefined simulation environment with materials and boundaries
        :param start_minimized: determines whether to start the FEMM window minimized. Starting it minimized speeds up the simulation
        :param dt: length of the time step of the simulation. The shorter, the better precission, but the longer it takes to simulate
        :param t_max: maximum time for which the simulation is allowed to run.
        By default, the simulation stops earlier, when the projectile velocity no longer changes from one frame to another
        """
        self._unique_femm_file_path = FEMMModel._init_femm(femm_file)
        self._circuit_model: circuit_model = circuit_model
        self._params = constraints
        self._dt = dt
        self._t_max = t_max

        # plots
        self._times = []
        self._velocities = []
        self._energies = []

        # defined in the .fem file
        self._projectile_side_group = 1
        self._projectile_base_group = 4
        self._projectile_tip_group = 6
        self._coil_outer_wall_group = 2
        self._coil_inner_wall_group = 5
        self._coil_height_group = 3

        self._initial_coil_size = 10  # the coil is initially defined as 10x10mm square
        self._initial_projectile_radius = (
            5  # the coil is initially defined as 10x10mm square
        )
        self._initial_projectile_length = self._initial_coil_size

        if start_minimized:
            femm.main_minimize()

    def _center_projectile(self, coil_length: float):
        femm.mi_selectgroup(self._projectile_tip_group)
        femm.mi_movetranslate(0, -coil_length / 2)
        femm.mi_selectgroup(self._projectile_base_group)
        femm.mi_movetranslate(0, -coil_length / 2)

    def _set_projectile_length(self, length: float):
        femm.mi_selectgroup(self._projectile_base_group)
        femm.mi_movetranslate(0, -length + self._initial_projectile_length)

    def _set_projectile_diameter(self, diameter: float):
        translation = diameter / 2.0 - self._initial_projectile_radius
        femm.mi_selectgroup(self._projectile_side_group)
        femm.mi_movetranslate(translation, 0)
        femm.mi_selectgroup(self._coil_outer_wall_group)
        femm.mi_movetranslate(translation, 0)
        femm.mi_selectgroup(self._coil_inner_wall_group)
        femm.mi_movetranslate(translation, 0)

    def _set_coil_size(self, length, layers):
        thickness = layers * self._params.wire_with_insulation_thickness_mm
        femm.mi_selectgroup(self._coil_outer_wall_group)
        femm.mi_movetranslate(thickness - self._initial_coil_size, 0)

        femm.mi_selectgroup(self._coil_height_group)
        femm.mi_movetranslate(0, -(length - self._initial_coil_size))

    def _set_coil_current_density(
        self, current_amps, axial_turns, coil_layers, coil_length
    ):
        amp_turns = axial_turns * coil_layers * current_amps
        winding_thickness = coil_layers * self._params.wire_with_insulation_thickness_mm
        area = winding_thickness * coil_length
        density = amp_turns / area
        current_density_property_number = 4  # see the femm 4.2 manual, page 90
        femm.mi_modifymaterial("Copper", current_density_property_number, density)

    def _calculate_projectile_mass_kg(self) -> float:
        area = pi * (self._params.projectile_diameter_mm / 2.0) ** 2
        volume_mm3 = self._params.projectile_length_mm * area
        return self._params.projectile_density_g_mm3 * volume_mm3 / 1000.0

    def _check_stop_criterion(self, tolerance=1e-6):
        current_energy = self._energies[len(self._energies) - 1]
        previous_energy = self._energies[len(self._energies) - 2]
        return abs(current_energy - previous_energy) < tolerance * current_energy

    def simulate(self) -> SimulationResult:
        """
        Runs FEMM simulation in conjunction with LTSPICE and COIL32 and returns exit kinetic energy of the projectile
        :return: SimulationResult dataclass which includes the plots and exit kinetic energy and velocity
        """
        self._set_coil_size(self._params.coil_length_mm, self._params.coil_layers_count)
        self._set_projectile_length(self._params.projectile_length_mm)
        self._set_projectile_diameter(self._params.projectile_diameter_mm)
        self._center_projectile(self._params.coil_length_mm)
        self._move_projectile(-self._params.projectile_initial_position_mm)
        current_velocity = self._params.initial_velocity_m_s

        self._compute_coil_current()

        for t in np.arange(0, self._t_max, self._dt):
            femm.mi_analyze()
            femm.mi_loadsolution()

            axial_turns = int(
                round(
                    self._params.coil_length_mm
                    / self._params.wire_with_insulation_thickness_mm
                )
            )
            self._set_coil_current_density(
                self._circuit_model.get_coil_current_in_time(t),
                axial_turns,
                self._params.coil_layers_count,
                self._params.coil_length_mm,
            )

            femm.mo_groupselectblock(self._projectile_tip_group)
            femm.mo_groupselectblock(self._projectile_base_group)
            fz = femm.mo_blockintegral(19)

            current_velocity += fz / self._calculate_projectile_mass_kg() * self._dt
            self._times.append(t)
            self._velocities.append(current_velocity)
            self._energies.append(
                current_velocity ** 2
                * self._calculate_projectile_mass_kg()
                * 0.5
            )

            #if self._check_stop_criterion():
                # Change in velocity below threshold. No point in computing further.
            #    break

            # double integrate acceleration over time
            displacement = current_velocity * self._dt * 1000
            self._move_projectile(displacement)

        femm.closefemm()
        cleanup(self._unique_femm_file_path.with_suffix(''))
        return SimulationResult(
            self._velocities[-1],
            self._energies[-1],
            copy(self._velocities),
            copy(self._energies),
            copy(self._times),
        )

    def _move_projectile(self, displacement):
        femm.mi_selectgroup(self._projectile_tip_group)
        femm.mi_movetranslate(0, displacement)
        femm.mi_selectgroup(self._projectile_base_group)
        femm.mi_movetranslate(0, displacement)

    def _compute_coil_current(self):
        coil_parameters = calc_multi_layer_for_known_layer_count(
            layers_count=self._params.coil_layers_count,
            inner_diameter=self._params.coil_inner_diameter_mm,
            length=self._params.coil_length_mm,
            wire_with_insulation_thickness=self._params.wire_with_insulation_thickness_mm,
            wire_without_insulation_thickness=self._params.wire_without_insulation_thickness_mm,
        )

        circuit_parameters = CircuitParameters(
            inductance_uH=coil_parameters.inductance_microhenries,
            coil_ESR_Ohm=coil_parameters.DC_resistance_ohms,
            capacitance_uF=self._params.capacitance_uF,
            cap_ESR_Ohm=self._params.capacitor_ESR_ohm,
        )

        self._circuit_model.simulate(circuit_parameters)
