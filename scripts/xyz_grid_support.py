from pathlib import Path

from modules import scripts
from scripts.dynamic_thresholding import VALID_MODES as scheduler_list


def find_module(name_list):
    if isinstance(name_list, str):
        name_list = [s.strip() for s in name_list.split(",")]

    for data in scripts.scripts_data:
        if Path(data.path).name in name_list:
            return data.module

    return None


def make_axis_options(xyz_grid):
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
            if x not in scheduler_list:
                raise RuntimeError(f"Unknown Scheduler: {x}")

            setattr(p, field, x)

        return core

    extra_axis_options = [
        AxisOption("DT Mimic Scale", float, apply_mimic_scale()),
        AxisOption("DT Threshold", float, apply_field("dynthres_threshold_percentile")),
        AxisOption("DT Mimic Scheduler", str, apply_scheduler("dynthres_mimic_mode"), choices=lambda: scheduler_list),
        AxisOption("DT Mimic min", float, apply_field("dynthres_mimic_scale_min")),
        AxisOption("DT CFG Scheduler", str, apply_scheduler("dynthres_cfg_mode"), choices=lambda: scheduler_list),
        AxisOption("DT CFG min", float, apply_field("dynthres_cfg_scale_min")),
        AxisOption("DT Power val", float, apply_field("dynthres_power_val")),
        AxisOption("DT Experiment", int, apply_field("dynthres_experiment_mode")),
    ]

    return extra_axis_options


def add_axis_options(axis_options, extra_axis_options, target_label=""):
    if target_label:
        for i, axis_option in enumerate(axis_options):
            if axis_option.label == target_label:
                axis_options[i+1:i+1] = extra_axis_options
                return

    axis_options.extend(extra_axis_options)
    return


xyz_grid = find_module("xyz_grid.py, xy_grid.py")

if xyz_grid:
    extra_axis_options = make_axis_options(xyz_grid)
    add_axis_options(xyz_grid.axis_options, extra_axis_options, "CFG Scale")