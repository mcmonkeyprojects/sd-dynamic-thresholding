##################
# Stable Diffusion Dynamic Thresholding (CFG Scale Fix)
#
# Author: Alex 'mcmonkey' Goodwin
# GitHub URL: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
# Created: 2022/01/26
# Last updated: 2023/01/30
#
# For usage help, view the README.md file in the extension root, or via the GitHub page.
#
##################

import gradio as gr
import torch, traceback
import dynthres_core
from modules import scripts, script_callbacks, sd_samplers, sd_samplers_compvis, sd_samplers_common
try:
    import dynthres_unipc
except Exception as e:
    print(f"\n\n======\nError! UniPC sampler support failed to load! Is your WebUI up to date?\n(Error: {e})\n======")
try:
    from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion as cfgdenoisekdiff
    IS_AUTO_16 = True
except Exception as e:
    print(f"\n\n======\nWarning! Using legacy KDiff version! Is your WebUI up to date?\n======")
    from modules.sd_samplers_kdiffusion import CFGDenoiser as cfgdenoisekdiff
    IS_AUTO_16 = False

DISABLE_VISIBILITY = True

######################### Data values #########################
MODES_WITH_VALUE = ["Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth"]

######################### Script class entrypoint #########################
class Script(scripts.Script):

    def title(self):
        return "Dynamic Thresholding (CFG Scale Fix)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        def vis_change(is_vis):
            return {"visible": is_vis, "__type__": "update"}
        # "Dynamic Thresholding (CFG Scale Fix)"
        dtrue = gr.Checkbox(value=True, visible=False)
        dfalse = gr.Checkbox(value=False, visible=False)
        with gr.Accordion("Dynamic Thresholding (CFG Scale Fix)", open=False, elem_id="dynthres_" + ("img2img" if is_img2img else "txt2img")):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label="Enable Dynamic Thresholding (CFG Scale Fix)", elem_classes=["dynthres-enabled"], elem_id='dynthres_enabled')
            with gr.Group():
                gr.HTML(value=f"View <a style=\"border-bottom: 1px #00ffff dotted;\" href=\"https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/wiki/Usage-Tips\">the wiki for usage tips.</a><br><br>", elem_id='dynthres_wiki_link')
                mimic_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Mimic CFG Scale', value=7.0, elem_id='dynthres_mimic_scale')
                with gr.Accordion("Advanced Options", open=False, elem_id='dynthres_advanced_opts'):
                    with gr.Row():
                        threshold_percentile = gr.Slider(minimum=90.0, value=100.0, maximum=100.0, step=0.05, label='Top percentile of latents to clamp', elem_id='dynthres_threshold_percentile')
                        interpolate_phi = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Interpolate Phi", value=1.0, elem_id='dynthres_interpolate_phi')
                    with gr.Row():
                        mimic_mode = gr.Dropdown(dynthres_core.DynThresh.Modes, value="Constant", label="Mimic Scale Scheduler", elem_id='dynthres_mimic_mode')
                        cfg_mode = gr.Dropdown(dynthres_core.DynThresh.Modes, value="Constant", label="CFG Scale Scheduler", elem_id='dynthres_cfg_mode')
                    mimic_scale_min = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, visible=DISABLE_VISIBILITY, label="Minimum value of the Mimic Scale Scheduler", elem_id='dynthres_mimic_scale_min')
                    cfg_scale_min = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, visible=DISABLE_VISIBILITY, label="Minimum value of the CFG Scale Scheduler", elem_id='dynthres_cfg_scale_min')
                    sched_val = gr.Slider(minimum=0.0, maximum=40.0, step=0.5, value=4.0, visible=DISABLE_VISIBILITY, label="Scheduler Value", info="Value unique to the scheduler mode - for Power Up/Down, this is the power. For Linear/Cosine Repeating, this is the number of repeats per image.", elem_id='dynthres_sched_val')
                    with gr.Row():
                        separate_feature_channels = gr.Checkbox(value=True, label="Separate Feature Channels", elem_id='dynthres_separate_feature_channels')
                        scaling_startpoint = gr.Radio(["ZERO", "MEAN"], value="MEAN", label="Scaling Startpoint")
                        variability_measure = gr.Radio(["STD", "AD"], value="AD", label="Variability Measure")
        def should_show_scheduler_value(cfg_mode, mimic_mode):
            sched_vis = cfg_mode in MODES_WITH_VALUE or mimic_mode in MODES_WITH_VALUE or DISABLE_VISIBILITY
            return vis_change(sched_vis), vis_change(mimic_mode != "Constant" or DISABLE_VISIBILITY), vis_change(cfg_mode != "Constant" or DISABLE_VISIBILITY)
        cfg_mode.change(should_show_scheduler_value, inputs=[cfg_mode, mimic_mode], outputs=[sched_val, mimic_scale_min, cfg_scale_min])
        mimic_mode.change(should_show_scheduler_value, inputs=[cfg_mode, mimic_mode], outputs=[sched_val, mimic_scale_min, cfg_scale_min])
        enabled.change(
            _js="dynthres_update_enabled",
            fn=None,
            inputs=[enabled, dtrue if is_img2img else dfalse],
            show_progress = False)
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value="Dynamic thresholding enabled" in d)),
            (mimic_scale, "Mimic scale"),
            (separate_feature_channels, "Separate Feature Channels"),
            (scaling_startpoint, lambda d: gr.Radio.update(value=d.get("Scaling Startpoint", "MEAN"))),
            (variability_measure, lambda d: gr.Radio.update(value=d.get("Variability Measure", "AD"))),
            (interpolate_phi, "Interpolate Phi"),
            (threshold_percentile, "Threshold percentile"),
            (mimic_scale_min, "Mimic scale minimum"),
            (mimic_mode, lambda d: gr.Dropdown.update(value=d.get("Mimic mode", "Constant"))),
            (cfg_mode, lambda d: gr.Dropdown.update(value=d.get("CFG mode", "Constant"))),
            (cfg_scale_min, "CFG scale minimum"),
            (sched_val, "Scheduler value"))
        return [enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi]

    last_id = 0

    def process_batch(self, p, enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi, batch_number, prompts, seeds, subseeds):
        enabled = getattr(p, 'dynthres_enabled', enabled)
        if not enabled:
            return
        orig_sampler_name = p.sampler_name
        orig_latent_sampler_name = getattr(p, 'latent_sampler', None)
        if orig_sampler_name in ["DDIM", "PLMS"]:
            raise RuntimeError(f"Cannot use sampler {orig_sampler_name} with Dynamic Thresholding")
        if orig_latent_sampler_name in ["DDIM", "PLMS"]:
            raise RuntimeError(f"Cannot use secondary sampler {orig_latent_sampler_name} with Dynamic Thresholding")
        if 'UniPC' in (orig_sampler_name, orig_latent_sampler_name) and p.enable_hr:
            raise RuntimeError(f"UniPC does not support Hires Fix. Auto WebUI silently swaps to DDIM for this, which DynThresh does not support. Please swap to a sampler capable of img2img processing for HR Fix to work.")
        mimic_scale = getattr(p, 'dynthres_mimic_scale', mimic_scale)
        separate_feature_channels = getattr(p, 'dynthres_separate_feature_channels', separate_feature_channels)
        scaling_startpoint = getattr(p, 'dynthres_scaling_startpoint', scaling_startpoint)
        variability_measure = getattr(p, 'dynthres_variability_measure', variability_measure)
        interpolate_phi = getattr(p, 'dynthres_interpolate_phi', interpolate_phi)
        threshold_percentile = getattr(p, 'dynthres_threshold_percentile', threshold_percentile)
        mimic_mode = getattr(p, 'dynthres_mimic_mode', mimic_mode)
        mimic_scale_min = getattr(p, 'dynthres_mimic_scale_min', mimic_scale_min)
        cfg_mode = getattr(p, 'dynthres_cfg_mode', cfg_mode)
        cfg_scale_min = getattr(p, 'dynthres_cfg_scale_min', cfg_scale_min)
        experiment_mode = getattr(p, 'dynthres_experiment_mode', 0)
        sched_val = getattr(p, 'dynthres_scheduler_val', sched_val)
        p.extra_generation_params["Dynamic thresholding enabled"] = True
        p.extra_generation_params["Mimic scale"] = mimic_scale
        p.extra_generation_params["Separate Feature Channels"] = separate_feature_channels
        p.extra_generation_params["Scaling Startpoint"] = scaling_startpoint
        p.extra_generation_params["Variability Measure"] = variability_measure
        p.extra_generation_params["Interpolate Phi"] = interpolate_phi
        p.extra_generation_params["Threshold percentile"] = threshold_percentile
        p.extra_generation_params["Sampler"] = orig_sampler_name
        if mimic_mode != "Constant":
            p.extra_generation_params["Mimic mode"] = mimic_mode
            p.extra_generation_params["Mimic scale minimum"] = mimic_scale_min
        if cfg_mode != "Constant":
            p.extra_generation_params["CFG mode"] = cfg_mode
            p.extra_generation_params["CFG scale minimum"] = cfg_scale_min
        if cfg_mode in MODES_WITH_VALUE or mimic_mode in MODES_WITH_VALUE:
            p.extra_generation_params["Scheduler value"] = sched_val
        # Note: the ID number is to protect the edge case of multiple simultaneous runs with different settings
        Script.last_id += 1
        # Percentage to portion
        threshold_percentile *= 0.01

        def make_sampler(orig_sampler_name):
            fixed_sampler_name = f"{orig_sampler_name}_dynthres{Script.last_id}"

            # Make a placeholder sampler
            sampler = sd_samplers.all_samplers_map[orig_sampler_name]
            dt_data = dynthres_core.DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, experiment_mode, p.steps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi)
            if orig_sampler_name == "UniPC":
                def unipc_constructor(model):
                    return CustomVanillaSDSampler(dynthres_unipc.CustomUniPCSampler, model, dt_data)
                new_sampler = sd_samplers_common.SamplerData(fixed_sampler_name, unipc_constructor, sampler.aliases, sampler.options)
            else:
                def new_constructor(model):
                    result = sampler.constructor(model)
                    cfg = CustomCFGDenoiser(result if IS_AUTO_16 else result.model_wrap_cfg.inner_model, dt_data)
                    result.model_wrap_cfg = cfg
                    return result
                new_sampler = sd_samplers_common.SamplerData(fixed_sampler_name, new_constructor, sampler.aliases, sampler.options)
            return fixed_sampler_name, new_sampler

        # Apply for usage
        p.orig_sampler_name = orig_sampler_name
        p.orig_latent_sampler_name = orig_latent_sampler_name
        p.fixed_samplers = []

        if orig_latent_sampler_name:
            latent_sampler_name, latent_sampler = make_sampler(orig_latent_sampler_name)
            sd_samplers.all_samplers_map[latent_sampler_name] = latent_sampler
            p.fixed_samplers.append(latent_sampler_name)
            p.latent_sampler = latent_sampler_name

        if orig_sampler_name != orig_latent_sampler_name:
            p.sampler_name, new_sampler = make_sampler(orig_sampler_name)
            sd_samplers.all_samplers_map[p.sampler_name] = new_sampler
            p.fixed_samplers.append(p.sampler_name)
        else:
            p.sampler_name = p.latent_sampler

        if p.sampler is not None:
            p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)

    def postprocess_batch(self, p, enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi, batch_number, images):
        if not enabled or not hasattr(p, 'orig_sampler_name'):
            return
        p.sampler_name = p.orig_sampler_name
        if p.orig_latent_sampler_name:
            p.latent_sampler = p.orig_latent_sampler_name
        for added_sampler in p.fixed_samplers:
            del sd_samplers.all_samplers_map[added_sampler]
        del p.fixed_samplers
        del p.orig_sampler_name
        del p.orig_latent_sampler_name

######################### CompVis Implementation logic #########################

class CustomVanillaSDSampler(sd_samplers_compvis.VanillaStableDiffusionSampler):
    def __init__(self, constructor, sd_model, dt_data):
        super().__init__(constructor, sd_model)
        self.sampler.main_class = dt_data

######################### K-Diffusion Implementation logic #########################

class CustomCFGDenoiser(cfgdenoisekdiff):
    def __init__(self, model, dt_data):
        super().__init__(model)
        self.main_class = dt_data

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        if isinstance(uncond, dict) and 'crossattn' in uncond:
            uncond = uncond['crossattn']
        denoised_uncond = x_out[-uncond.shape[0]:]
        # conds_list shape is (batch, cond, 2)
        weights = torch.tensor(conds_list, device=uncond.device).select(2, 1)
        weights = weights.reshape(*weights.shape, 1, 1, 1)
        self.main_class.step = self.step
        if hasattr(self, 'total_steps'):
            self.main_class.max_steps = self.total_steps

        if self.main_class.experiment_mode >= 4 and self.main_class.experiment_mode <= 5:
            # https://arxiv.org/pdf/2305.08891.pdf "Rescale CFG". It's not good, but if you want to test it, just set experiment_mode = 4 + phi.
            denoised = torch.clone(denoised_uncond)
            fi = self.main_class.experiment_mode - 4.0
            for i, conds in enumerate(conds_list):
                for cond_index, weight in conds:
                    xcfg = (denoised_uncond[i] + (x_out[cond_index] - denoised_uncond[i]) * (cond_scale * weight))
                    xrescaled = xcfg * (torch.std(x_out[cond_index]) / torch.std(xcfg))
                    xfinal = fi * xrescaled + (1.0 - fi) * xcfg
                    denoised[i] = xfinal
            return denoised

        return self.main_class.dynthresh(x_out[:-uncond.shape[0]], denoised_uncond, cond_scale, weights)

######################### XYZ Plot Script Support logic #########################

def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
    def apply_mimic_scale(p, x, xs):
        if x != 0:
            setattr(p, "dynthres_enabled", True)
            setattr(p, "dynthres_mimic_scale", x)
        else:
            setattr(p, "dynthres_enabled", False)
    def confirm_scheduler(p, xs):
        for x in xs:
            if x not in dynthres_core.DynThresh.Modes:
                raise RuntimeError(f"Unknown Scheduler: {x}")
    extra_axis_options = [
        xyz_grid.AxisOption("[DynThres] Mimic Scale", float, apply_mimic_scale),
        xyz_grid.AxisOption("[DynThres] Separate Feature Channels", int,
                            xyz_grid.apply_field("dynthres_separate_feature_channels")),
        xyz_grid.AxisOption("[DynThres] Scaling Startpoint", str, xyz_grid.apply_field("dynthres_scaling_startpoint"), choices=lambda:['ZERO', 'MEAN']),
        xyz_grid.AxisOption("[DynThres] Variability Measure", str, xyz_grid.apply_field("dynthres_variability_measure"), choices=lambda:['STD', 'AD']),
        xyz_grid.AxisOption("[DynThres] Interpolate Phi", float, xyz_grid.apply_field("dynthres_interpolate_phi")),
        xyz_grid.AxisOption("[DynThres] Threshold Percentile", float, xyz_grid.apply_field("dynthres_threshold_percentile")),
        xyz_grid.AxisOption("[DynThres] Mimic Scheduler", str, xyz_grid.apply_field("dynthres_mimic_mode"), confirm=confirm_scheduler, choices=lambda: dynthres_core.DynThresh.Modes),
        xyz_grid.AxisOption("[DynThres] Mimic minimum", float, xyz_grid.apply_field("dynthres_mimic_scale_min")),
        xyz_grid.AxisOption("[DynThres] CFG Scheduler", str, xyz_grid.apply_field("dynthres_cfg_mode"), confirm=confirm_scheduler, choices=lambda: dynthres_core.DynThresh.Modes),
        xyz_grid.AxisOption("[DynThres] CFG minimum", float, xyz_grid.apply_field("dynthres_cfg_scale_min")),
        xyz_grid.AxisOption("[DynThres] Scheduler value", float, xyz_grid.apply_field("dynthres_scheduler_val"))
    ]
    if not any("[DynThres]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
    try:
        make_axis_options()
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to add support for X/Y/Z Plot Script because: {e}")

script_callbacks.on_before_ui(callback_before_ui)
