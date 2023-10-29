from .dynthres_core import DynThresh

class DynamicThresholdingComfyNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mimic_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mimic_mode": (DynThresh.Modes, ),
                "mimic_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_mode": (DynThresh.Modes, ),
                "cfg_scale_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sched_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "separate_feature_channels": (["enable", "disable"], ),
                "scaling_startpoint": (DynThresh.Startpoints, ),
                "variability_measure": (DynThresh.Variabilities, ),
                "interpolate_phi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/mcmonkey"

    def patch(self, model, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):

        dynamic_thresh = DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, 0, 999, separate_feature_channels == "enable", scaling_startpoint, variability_measure, interpolate_phi)
        
        def sampler_dyn_thrash(args):
            x_out = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return dynamic_thresh.dynthresh(x_out, uncond, cond_scale, None)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thrash)
        return (m, )

class DynamicThresholdingSimpleComfyNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mimic_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/mcmonkey"

    def patch(self, model, mimic_scale, threshold_percentile):

        dynamic_thresh = DynThresh(mimic_scale, threshold_percentile, "CONSTANT", 0, "CONSTANT", 0, 0, 0, 999, False, "MEAN", "AD", 1)
        
        def sampler_dyn_thrash(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thrash)
        return (m, )
