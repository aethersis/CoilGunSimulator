import os
import shutil
import glob
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

from PyLTSpice.LTSpiceBatch import LTCommander
from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead

from src.utils import generate_unique_file_name, cleanup


@dataclass(frozen=True)
class CircuitParameters:
    inductance_uH: float
    capacitance_uF: float
    cap_ESR_Ohm: float
    coil_ESR_Ohm: float


def _get_correct_simulation_result_path(log_path: str):
    """
    Fixes a bug in PyLTSpice that causes the raw file path to be returned with invalid name. By default, it's always
    returned with _run.raw while it should be _<ID>.raw. The correct ID is contained in the log file path and it is
    used in this method to generate the correct raw file path.
    :param log_path: path to the log file
    :return: correct path to the simulation result file
    """
    stem = os.path.splitext(log_path)[0]
    return stem + ".raw"


class CircuitModel:
    def __init__(
        self, spice_file: str, history_file_path: Optional[str] = "circuit_history.pil"
    ):
        # Create a unique copy of the source spice file to avoid thread locking with multiple backend workers
        if not os.path.exists("tmp") and not os.path.isdir("tmp"):
            os.mkdir("tmp")
        self._spice_file_path = generate_unique_file_name("tmp", ".asc").as_posix()
        shutil.copy(spice_file, self._spice_file_path)

        self._model = LTCommander(self._spice_file_path)
        self._history_file_path = history_file_path

        self._coil_current = []
        self._coil_time = []
        self._history: Dict[CircuitParameters, Tuple] = {}

        if history_file_path is not None and os.path.exists(history_file_path):
            self._load_history(history_file_path)

        self._current_parameters = None

    def _load_history(self, history_file_path: str):
        with open(history_file_path, "rb") as fp:
            self._history = pickle.load(fp)

    def _save_history(self, history_file_path: str):
        with open(history_file_path, "wb") as fp:
            pickle.dump(self._history, fp)

    def _collect_results(self, raw: str, log: str):
        raw = raw.replace("\\", "/")
        parser = LTSpiceRawRead(raw)
        self._coil_current = parser.get_trace("I(L2)").data
        self._coil_time = parser.get_trace("I(L2)").axis.data

        if self._history_file_path is not None:
            self._history[self._current_parameters] = (
                self._coil_current,
                self._coil_time,
            )
            self._save_history(self._history_file_path)

    def get_coil_current_in_time(self, time_s) -> float:
        index = min(enumerate(self._coil_time), key=lambda x: abs(x[1] - time_s))
        return self._coil_current[index[0]]

    def simulate(self, parameters: CircuitParameters):
        self._model.set_component_value("L2", f"{parameters.inductance_uH}u")
        self._model.set_component_value("R2", parameters.coil_ESR_Ohm)

        self._model.set_component_value("C1", f"{parameters.capacitance_uF}u")
        self._model.set_component_value("R1", parameters.cap_ESR_Ohm)
        self._current_parameters = parameters

        if parameters not in self._history or self._history_file_path is None:
            raw, log = self._model.run()
            raw = _get_correct_simulation_result_path(log)
            self._collect_results(raw, log)
            cleanup(os.path.splitext(raw)[0])
        else:
            self._coil_current, self._coil_time = self._history[parameters]
        cleanup(Path(self._spice_file_path).with_suffix(''))
        cleanup('tmp/net' + str(parameters.__hash__()))
