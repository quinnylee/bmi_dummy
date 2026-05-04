"""
Dummy BMI Model
---------------
A no-op BMI model that accepts any input variables and always returns 0.0
for every output variable at every time step.

Compatible with the NextGen / NGIAB Basic Model Interface (BMI) standard.
"""

import numpy as np
import yaml


class BmiDummy:
    """Dummy BMI model — accepts anything, always outputs 0."""

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self, config_file: str = "") -> None:
        """
        Initialize the model from an optional YAML config file.

        The config file may define:
            - start_time   (float, default 0.0)
            - end_time     (float, default 86400.0  — one day in seconds)
            - time_step    (float, default 3600.0   — one hour in seconds)
            - input_vars   (list of str, default ["input"])
            - output_vars  (list of str, default ["output"])

        All values can also be omitted; defaults are used in that case.
        """
        # Sensible defaults
        self._start_time: float = 0.0
        self._end_time: float = 86400.0
        self._time_step: float = 3600.0
        self._input_var_names: list[str] = ["input"]
        self._output_var_names: list[str] = ["output"]

        if config_file:
            try:
                with open(config_file, "r") as f:
                    cfg = yaml.safe_load(f) or {}  # /dev/null yields None → {}
                self._start_time = float(cfg.get("start_time", self._start_time))
                self._end_time = float(cfg.get("end_time", self._end_time))
                self._time_step = float(cfg.get("time_step", self._time_step))
                if "input_vars" in cfg:
                    self._input_var_names = list(cfg["input_vars"])
                if "output_vars" in cfg:
                    self._output_var_names = list(cfg["output_vars"])
            except (FileNotFoundError, TypeError):
                pass  # no config or unparseable — use defaults

        self._current_time: float = self._start_time

        # Storage: one scalar (float64) per variable, always zero
        self._input_store: dict[str, np.ndarray] = {
            v: np.zeros(1, dtype=np.float64) for v in self._input_var_names
        }
        self._output_store: dict[str, np.ndarray] = {
            v: np.zeros(1, dtype=np.float64) for v in self._output_var_names
        }

    def update(self) -> None:
        """Advance the model by one time step. Outputs remain 0."""
        self._current_time += self._time_step

    def update_until(self, time: float) -> None:
        """Advance the model until *time*."""
        while self._current_time < time:
            self.update()

    def finalize(self) -> None:
        """Tear down the model (nothing to do for a dummy)."""
        pass

    # ------------------------------------------------------------------ #
    #  Model information                                                   #
    # ------------------------------------------------------------------ #

    def get_component_name(self) -> str:
        return "Dummy BMI Model"

    def get_input_item_count(self) -> int:
        return len(self._input_var_names)

    def get_output_item_count(self) -> int:
        return len(self._output_var_names)

    # ------------------------------------------------------------------ #
    #  Time                                                                #
    # ------------------------------------------------------------------ #

    def get_start_time(self) -> float:
        return self._start_time

    def get_end_time(self) -> float:
        return self._end_time

    def get_current_time(self) -> float:
        return self._current_time

    def get_time_step(self) -> float:
        return self._time_step

    def get_time_units(self) -> str:
        return "s"  # seconds

    # ------------------------------------------------------------------ #
    #  Variable names                                                      #
    # ------------------------------------------------------------------ #

    def get_input_var_names(self) -> tuple[str, ...]:
        return tuple(self._input_var_names)

    def get_output_var_names(self) -> tuple[str, ...]:
        return tuple(self._output_var_names)

    # ------------------------------------------------------------------ #
    #  Variable metadata                                                   #
    # ------------------------------------------------------------------ #

    def get_var_type(self, name: str) -> str:
        return "float64"

    def get_var_units(self, name: str) -> str:
        return "1"  # dimensionless

    def get_var_itemsize(self, name: str) -> int:
        return np.dtype("float64").itemsize  # 8 bytes

    def get_var_nbytes(self, name: str) -> int:
        return self.get_var_itemsize(name) * self.get_var_grid_size(
            self.get_var_grid(name)
        )

    def get_var_location(self, name: str) -> str:
        return "node"

    def get_var_grid(self, name: str) -> int:
        return 0  # single scalar grid

    # ------------------------------------------------------------------ #
    #  Grid metadata (single scalar point, grid id = 0)                   #
    # ------------------------------------------------------------------ #

    def get_grid_rank(self, grid: int) -> int:
        return 1

    def get_grid_size(self, grid: int) -> int:
        return 1

    def get_grid_type(self, grid: int) -> str:
        return "scalar"

    def get_grid_shape(self, grid: int, shape: np.ndarray) -> np.ndarray:
        shape[:] = [1]
        return shape

    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> np.ndarray:
        spacing[:] = [1.0]
        return spacing

    def get_grid_origin(self, grid: int, origin: np.ndarray) -> np.ndarray:
        origin[:] = [0.0]
        return origin

    def get_grid_node_count(self, grid: int) -> int:
        return 1

    def get_grid_edge_count(self, grid: int) -> int:
        return 0

    def get_grid_face_count(self, grid: int) -> int:
        return 0

    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        x[:] = [0.0]
        return x

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        y[:] = [0.0]
        return y

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        z[:] = [0.0]
        return z

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        return edge_nodes  # no edges on a scalar grid

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        return face_edges  # no faces on a scalar grid

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        return face_nodes  # no faces on a scalar grid

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: np.ndarray) -> np.ndarray:
        return nodes_per_face  # no faces on a scalar grid

    # ------------------------------------------------------------------ #
    #  Getters / Setters                                                   #
    # ------------------------------------------------------------------ #

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """Copy the current value (always 0) into *dest*."""
        dest[:] = self._output_store.get(
            name, self._input_store.get(name, np.zeros(1, dtype=np.float64))
        )
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Return a direct reference to the internal array (always 0)."""
        if name in self._output_store:
            return self._output_store[name]
        return self._input_store[name]

    def get_value_at_indices(
        self, name: str, dest: np.ndarray, inds: np.ndarray
    ) -> np.ndarray:
        dest[:] = 0.0
        return dest

    def set_value(self, name: str, src: np.ndarray) -> None:
        """Accept any input value (stored but never used in computation)."""
        if name in self._input_store:
            self._input_store[name][:] = src
        # Unknown variable names are silently accepted

    def set_value_at_indices(
        self, name: str, inds: np.ndarray, src: np.ndarray
    ) -> None:
        if name in self._input_store:
            self._input_store[name][inds] = src


# --------------------------------------------------------------------------- #
#  Quick smoke test                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    model = BmiDummy()
    model.initialize()  # no config file — uses defaults

    print(f"Model   : {model.get_component_name()}")
    print(f"Inputs  : {model.get_input_var_names()}")
    print(f"Outputs : {model.get_output_var_names()}")
    print(f"Start   : {model.get_start_time()} {model.get_time_units()}")
    print(f"End     : {model.get_end_time()} {model.get_time_units()}")
    print(f"dt      : {model.get_time_step()} {model.get_time_units()}")
    print()

    dest = np.zeros(1, dtype=np.float64)
    while model.get_current_time() < model.get_end_time():
        # Pretend we received some forcing data
        model.set_value("input", np.array([999.9]))

        model.update()

        model.get_value("output", dest)
        print(f"  t={model.get_current_time():8.1f} s  |  output = {dest[0]}")

    model.finalize()
    print("\nDone.")
