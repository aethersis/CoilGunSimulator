from copy import copy
from multiprocessing.dummy import freeze_support

from src.femm_model import SimulationParameters, FEMMModel, SimulationResult
from src.spice_model import CircuitModel
import matplotlib.pyplot as plt
from multiprocessing import Pool


base_params = SimulationParameters(
    coil_inner_diameter_mm=8,
    projectile_diameter_mm=6.5,
    wire_without_insulation_thickness_mm=1.0,
    wire_with_insulation_thickness_mm=1.07,
    capacitance_uF=500.0,
    capacitor_ESR_ohm=0.09,
    capacitor_voltage_V=370,
    coil_length_mm=24,
    coil_layers_count=5,
    projectile_length_mm=30,
    projectile_initial_position_mm=8,
)


circuit_model = CircuitModel("models/spice/thyristor_coilgun.asc")


def plot_optimization_results(results, x_property_name, y_property_name="end_kinetic_energy_J"):
    x = []
    y = []
    for value in results:
        input: SimulationParameters = value[0]
        result: SimulationResult = value[1]

        x.append(getattr(input, x_property_name))
        y.append(getattr(result, y_property_name))

    plt.plot(x, y)
    plt.xlabel(x_property_name)
    plt.ylabel(y_property_name)
    plt.show()


def plot_energy(results: SimulationResult):
    plt.plot(results.timescale, results.energy_plot)
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.show()


def plot_velocity(results: SimulationResult):
    plt.plot(results.timescale, results.velocity_plot)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.show()


def run_femm(params: SimulationParameters):
    s = FEMMModel(constraints=params, circuit_model=circuit_model, femm_file="models/femm/coilgun.fem")
    return params, s.simulate()


def run_optimizer():
    sweep = []
    freeze_support()
    for x in range(1, 24, 1):
        new_params = copy(base_params)
        new_params.projectile_initial_position_mm = x
        sweep.append(new_params)

    thread_pool = Pool(10)
    results = thread_pool.map(run_femm, sweep)
    plot_optimization_results(results, 'projectile_initial_position_mm')


if __name__ == '__main__':
    ans = run_femm(base_params)
    plot_energy(ans[1])
    plot_velocity(ans[1])