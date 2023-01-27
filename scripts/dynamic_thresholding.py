##################
# Stable Diffusion Dynamic Thresholding (CFG Scale Fix)
#
# Author: Alex 'mcmonkey' Goodwin
# GitHub URL: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
# Created: 2022/01/26
# Last updated: 2023/01/26
#
# For usage help, view the README.md file in the extension root, or via the GitHub page.
#
##################

import gradio as gr
import random
import torch
import math
from modules import sd_samplers, scripts

######################### Data values #########################
VALID_MODES = ["Constant", "Linear Down", "Cosine Down", "Linear Up", "Cosine Up"]

######################### Script class entrypoint #########################
class Script(scripts.Script):

    def title(self):
        return "Dynamic Thresholding (CFG Scale Fix)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enabled = gr.Checkbox(value=False, label="Enable Dynamic Thresholding (CFG Scale Fix)")
        # "Dynamic Thresholding (CFG Scale Fix)"
        accordion = gr.Group(visible=False)
        with accordion:
            gr.Markdown("Thresholds high CFG scales to make them work better.  \nSet your actual **CFG Scale** to the high value you want above (eg: 20).  \nThen set '**Mimic CFG Scale**' below to a (lower) CFG scale to mimic the effects of (eg: 10). Make sure it's not *too* different from your actual scale, it can only compensate so far.  \nSet '**Top percentile**' to how much clamping you want. 90% is good is normal, 100% clamps so hard it's like the mimic scale is the real scale. This scales as it approaches 100%, (eg 90% and 95% are much more similar than 98% and 99%).  \n...  \n")
            mimic_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Mimic CFG Scale', value=7.0)
            threshold_percentile = gr.Slider(minimum=90.0, value=90.0, maximum=100.0, step=0.05, label='Top percentile of latents to clamp')
            with gr.Accordion("Dynamic Thresholding Advanced Options", open=False):
                gr.Markdown("You can configure the **scale scheduler** for either the CFG Scale or the Mimic Scale here.  \n'**Constant**' is normal.  \nSetting **Mimic** to '**Cosine Down**' seems to produce better results. Needs more testing.  \nSetting **CFG** to '**Linear Down**' produces results that are just like the raw high scale CFG but with better quality fine details.  \nOther setting combos produce interesting results as well.  \n... \n")
                mimic_mode = gr.Dropdown(VALID_MODES, value="Constant", label="Mimic Scale Scheduler")
                cfg_mode = gr.Dropdown(VALID_MODES, value="Constant", label="CFG Scale Scheduler")
        enabled.change(
            fn=lambda x: {"visible": x, "__type__": "update"},
            inputs=[enabled],
            outputs=[accordion],
            show_progress = False)
        return [enabled, mimic_scale, threshold_percentile, mimic_mode, cfg_mode]

    def process_batch(self, p, enabled, mimic_scale, threshold_percentile, mimic_mode, cfg_mode, batch_number, prompts, seeds, subseeds):
        enabled = p.dynthres_enabled if hasattr(p, 'dynthres_enabled') else enabled
        if not enabled:
            return
        if p.sampler_name in ["DDIM", "PLMS"]:
            raise RuntimeError(f"Cannot use sampler {p.sampler_name} with Dynamic Thresholding")
        mimic_scale = p.dynthres_mimic_scale if hasattr(p, 'dynthres_mimic_scale') else mimic_scale
        threshold_percentile = p.dynthres_threshold_percentile if hasattr(p, 'dynthres_threshold_percentile') else threshold_percentile
        mimic_mode = p.dynthres_mimic_mode if hasattr(p, 'dynthres_mimic_mode') else mimic_mode
        cfg_mode = p.dynthres_cfg_mode if hasattr(p, 'dynthres_cfg_mode') else cfg_mode
        # Note: the random number is to protect the edge case of multiple simultaneous runs with different settings
        fixed_sampler_name = f"{p.sampler_name}_dynthres{random.randrange(100)}"
        p.fixed_sampler_name = fixed_sampler_name
        # Percentage to portion
        threshold_percentile *= 0.01
        # Make a placeholder sampler
        sampler = sd_samplers.all_samplers_map[p.sampler_name]
        def newConstructor(model):
            result = sampler.constructor(model)
            cfg = CustomCFGDenoiser(result.model_wrap_cfg.inner_model, mimic_scale, threshold_percentile, mimic_mode, cfg_mode, p.steps)
            result.model_wrap_cfg = cfg
            return result
        newSampler = sd_samplers.SamplerData(fixed_sampler_name, newConstructor, sampler.aliases, sampler.options)
        p.sampler_name = fixed_sampler_name
        sd_samplers.all_samplers_map[fixed_sampler_name] = newSampler

    def postprocess_batch(self, p, enabled, mimic_scale, threshold_percentile, mimic_mode, cfg_mode, batch_number, images):
        if not enabled:
            return
        del sd_samplers.all_samplers_map[p.fixed_sampler_name]

######################### Implementation logic #########################

class CustomCFGDenoiser(sd_samplers.CFGDenoiser):
    def __init__(self, model, mimic_scale, threshold_percentile, mimic_mode, cfg_mode, maxSteps):
        super().__init__(model)
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.maxSteps = maxSteps

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        return self.dynthresh(x_out[:-uncond.shape[0]], denoised_uncond, cond_scale, conds_list)

    def dynthresh(self, cond, uncond, cond_scale, conds_list):
        mimicScale = self.mimic_scale
        match self.mimic_mode:
            case "Constant":
                pass
            case "Linear Down":
                mimicScale *= 1.0 - (self.step / self.maxSteps)
            case "Cosine Down":
                mimicScale *= 1.0 - math.cos(self.step / self.maxSteps)
            case "Linear Up":
                mimicScale *= self.step / self.maxSteps
                pass
            case "Cosine Up":
                mimicScale *= math.cos(self.step / self.maxSteps)
                pass
        match self.cfg_mode:
            case "Constant":
                pass
            case "Linear Down":
                cond_scale *= 1.0 - (self.step / self.maxSteps)
            case "Cosine Down":
                cond_scale *= 1.0 - math.cos(self.step / self.maxSteps)
            case "Linear Up":
                cond_scale *= self.step / self.maxSteps
                pass
            case "Cosine Up":
                cond_scale *= math.cos(self.step / self.maxSteps)
                pass
        # uncond shape is (batch, 4, height, width)
        conds_per_batch = cond.shape[0] / uncond.shape[0]
        assert conds_per_batch == int(conds_per_batch), "Expected # of conds per batch to be constant across batches"
        cond_stacked = cond.reshape((-1, int(conds_per_batch)) + uncond.shape[1:])
        diff = cond_stacked - uncond.unsqueeze(1)
        # conds_list shape is (batch, cond, 2)
        weights = torch.tensor(conds_list).select(2, 1)
        weights = weights.reshape(*weights.shape, 1, 1, 1).to(diff.device)
        diff_weighted = (diff * weights).sum(1)
        dynthresh_target = uncond + diff_weighted * mimicScale

        dt_flattened = dynthresh_target.flatten(2)
        dt_means = dt_flattened.mean(dim=2).unsqueeze(2)
        dt_recentered = dt_flattened - dt_means
        dt_max = dt_recentered.abs().max(dim=2).values.unsqueeze(2)

        ut = uncond + diff_weighted * cond_scale
        ut_flattened = ut.flatten(2)
        ut_means = ut_flattened.mean(dim=2).unsqueeze(2)
        ut_centered = ut_flattened - ut_means

        ut_q = torch.quantile(ut_centered.abs(), self.threshold_percentile, dim=2).unsqueeze(2)
        s = torch.maximum(ut_q, dt_max)
        t_clamped = ut_centered.clamp(-s, s)
        t_normalized = t_clamped / s
        t_renormalized = t_normalized * dt_max

        uncentered = t_renormalized + ut_means
        unflattened = uncentered.unflatten(2, dynthresh_target.shape[2:])
        return unflattened
