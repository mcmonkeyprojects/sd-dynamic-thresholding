from pathlib import Path

from modules import scripts
from scripts.dynamic_thresholding import VALID_MODES


def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

    AxisOption = xyz_grid.AxisOption
    apply_field = xyz_grid.apply_field

    def apply_mimic_scale():
        def core(p, x, xs):
            if x != 0:
                setattr(p, "dynthres_enabled", True)
                setattr(p, "dynthres_mimic_scale", x)
            else:
                setattr(p, "dynthres_enabled", False)

        return core

    def apply_scheduler(field):
        def core(p, x, xs):
            if x not in VALID_MODES:
                raise RuntimeError(f"Unknown Scheduler: {x}")

            setattr(p, field, x)

        return core

    extra_axis_options = [
        AxisOption("DT Mimic Scale", float, apply_mimic_scale()),
        AxisOption("DT Threshold", float, apply_field("dynthres_threshold_percentile")),
        AxisOption("DT Mimic Scheduler", str, apply_scheduler("dynthres_mimic_mode"), choices=lambda: VALID_MODES),
        AxisOption("DT Mimic min", float, apply_field("dynthres_mimic_scale_min")),
        AxisOption("DT CFG Scheduler", str, apply_scheduler("dynthres_cfg_mode"), choices=lambda: VALID_MODES),
        AxisOption("DT CFG min", float, apply_field("dynthres_cfg_scale_min")),
        AxisOption("DT Power val", float, apply_field("dynthres_power_val")),
        AxisOption("DT Experiment", int, apply_field("dynthres_experiment_mode")),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)

make_axis_options()
