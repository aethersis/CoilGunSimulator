from copy import copy
from math import pi
from typing import List

import femm
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead


@dataclass
class SimulationParameters:
    inner_diameter_mm: float = 10.0
    projectile_diameter: float = 8.5
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


class SpiceSimulator:
    def __init__(self, constraints: SimulationParameters, search_ranges: Constraints, femm_file: str = "coilgun_9mm.fem",
                 start_minimized: bool = True, dt: float = 0.00005, t_max: float = 0.004):
        pass


class FEMMModel:
    @staticmethod
    def init_femm(femm_file: str):
        femm.openfemm()
        femm.opendocument(femm_file)
        femm.mi_saveas("temp.fem")
        femm.mi_seteditmode("group")
        femm.mi_analyze()
        femm.mi_loadsolution()

    def __init__(self, constraints: SimulationParameters, femm_file: str = "coilgun_metal_casing.fem",
                 start_minimized: bool = True, dt: float = 0.00005, t_max: float = 0.004):
        FEMMModel.init_femm(femm_file)

        self._params = constraints
        self._dt = dt
        self._t_max = t_max

        # plots
        self._times = []
        self._velocities = []
        self._energies = []

        # defined in the .fem file
        self._projectile_tip_side_group = 1
        self._projectile_base_group = 4
        self._coil_width_group = 2
        self._coil_height_group = 3
        self._initial_coil_size = 10  # the coil is initially defined as 10x10mm square
        self._initial_projectile_length = self._initial_coil_size

        if start_minimized:
            femm.main_minimize()

        LTR = LTSpiceRawRead("../Simulation.raw")
        self._coil_current = LTR.get_trace("I(L2)").data
        self._coil_time = LTR.get_trace("I(L2)").axis.data

    def _get_current_in_time(self, time) -> float:
        index = min(enumerate(self._coil_time), key=lambda x: abs(x[1] - time))
        return self._coil_current[index[0]]

    def _center_projectile(self, coil_length: float):
        femm.mi_selectgroup(self._projectile_tip_side_group)
        femm.mi_movetranslate(0, -coil_length / 2)

    def _set_projectile_length(self, length: float):
        femm.mi_selectgroup(self._projectile_base_group)
        femm.mi_movetranslate(0, -length + self._initial_projectile_length)

    def _set_coil_size(self, length, layers):
        thickness = layers * self._params.wire_with_insulation_thickness_mm
        femm.mi_selectgroup(self._coil_width_group)
        femm.mi_movetranslate(thickness - self._initial_coil_size, 0)

        femm.mi_selectgroup(self._coil_height_group)
        femm.mi_movetranslate(0, -(length - self._initial_coil_size))

    def _set_coil_current_density(self, current_amps, axial_turns, coil_layers, coil_length):
        amp_turns = axial_turns * coil_layers * current_amps
        winding_thickness = coil_layers * self._params.wire_with_insulation_thickness_mm
        area = winding_thickness * coil_length
        density = amp_turns / area
        current_density_property_number = 4  # see the femm 4.2 manual, page 90
        femm.mi_modifymaterial("Copper", current_density_property_number, density)

    def _calculate_projectile_mass_kg(self) -> float:
        area = pi * (self._params.projectile_diameter / 2.0) ** 2
        volume_mm3 = self._params.projectile_length_mm * area
        return self._params.projectile_density_g_mm3 * volume_mm3 / 1000.0

    def _check_stop_criterion(self, tollerance=1e-3):
        current_energy = self._energies[len(self._energies) - 1]
        previous_energy = self._energies[len(self._energies) - 2]
        return abs(current_energy - previous_energy) < tollerance * current_energy

    def simulate(self) -> SimulationResult:
        """
        Runs FEMM simulation in conjunction with LTSPICE and COIL32 and returns exit kinetic energy of the projectile
        :return: SimulationResult dataclass which includes the plots and exit kinetic energy and velocity
        """
        self._set_coil_size(self._params.coil_length_mm,
                            self._params.coil_layers_count)
        self._set_projectile_length(self._params.projectile_length_mm)
        self._center_projectile(self._params.coil_length_mm)
        self._move_projectile(-self._params.projectile_initial_position_mm)
        current_acceleration = 0.0

        for t in np.arange(0, self._t_max, self._dt):
            femm.mi_analyze()
            femm.mi_loadsolution()

            axial_turns = self._params.coil_length_mm  # TODO FIX THIS

            self._set_coil_current_density(self._get_current_in_time(t),
                                           axial_turns,
                                           self._params.coil_layers_count,
                                           self._params.coil_length_mm)

            femm.mo_groupselectblock(self._projectile_tip_side_group)
            femm.mo_groupselectblock(self._projectile_base_group)
            fz = femm.mo_blockintegral(19)

            current_acceleration += fz / self._calculate_projectile_mass_kg()
            self._times.append(t)
            self._velocities.append(current_acceleration * self._dt)
            self._energies.append((current_acceleration * self._dt) ** 2 * self._calculate_projectile_mass_kg() * 0.5)

            if self._check_stop_criterion():
                # Change in velocity below threshold. No point in computing further.
                break

            # double integrate acceleration over time
            displacement = current_acceleration * self._dt ** 2 * 1000
            self._move_projectile(displacement)

        femm.closefemm()
        return SimulationResult(self._velocities[-1], self._energies[-1], copy(self._velocities), copy(self._energies), copy(self._times))

    def _move_projectile(self, displacement):
        femm.mi_selectgroup(self._projectile_tip_side_group)
        femm.mi_movetranslate(0, displacement)


def plot_velocity(velocities: List[float], timescale: List[float]):
    plt.plot(timescale, velocities)
    plt.xlabel('Time, s')
    plt.ylabel('Velocity, m/s')
    plt.show()


def plot_energy(energies: List[float], timescale: List[float]):
    plt.plot(timescale, energies)
    plt.xlabel('Time, s')
    plt.ylabel('Energy, J')
    plt.show()


s = FEMMModel(SimulationParameters())
result = s.simulate()
print(result.end_kinetic_energy_J)
plot_energy(result.energy_plot, result.timescale)
plot_velocity(result.velocity_plot, result.timescale)