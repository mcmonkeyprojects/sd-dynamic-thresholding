import gradio as gr
import torch
import math
import traceback
from modules import shared
try:
    from modules.models.diffusion import uni_pc
except Exception as e:
    from modules import unipc as uni_pc

######################### UniPC Implementation logic #########################

# The majority of this is straight from modules.models/diffusion/uni_pc/sampler.py
# Unfortunately that's not an easy middle-injection point, so, just copypasta'd it all
# It's like they designed it to intentionally be as difficult to inject into as possible :(
# (It has hooks but not in useful locations)
# I stripped the original comments for brevity.
# Some never-used code (scheduler modes, noise modes, guidance modes) have been removed as well for brevity.
# The actual impl comes down to just the last line in particular, and the `before_sample` insert to track step count.

class CustomUniPCSampler(uni_pc.sampler.UniPCSampler):
    def __init__(self, model, **kwargs):
        super().__init__(model, *kwargs)
    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None,
               quantize_x0=False, eta=0., mask=None, x0=None, temperature=1., noise_dropout=0., score_corrector=None,
               corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1.,
               unconditional_conditioning=None, **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        C, H, W = shape
        size = (batch_size, C, H, W)
        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T
        ns = uni_pc.uni_pc.NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        model_type = "v" if self.model.parameterization == "v" else "noise"
        model_fn = CustomUniPC_model_wrapper(lambda x, t, c: self.model.apply_model(x, t, c), ns, model_type=model_type, guidance_scale=unconditional_guidance_scale, dt_data=self.main_class)
        self.main_class.step = 0
        def before_sample(x, t, cond, uncond):
            self.main_class.step += 1
            return self.before_sample(x, t, cond, uncond)
        uni_pc_inst = uni_pc.uni_pc.UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=shared.opts.uni_pc_variant, condition=conditioning, unconditional_condition=unconditional_conditioning, before_sample=before_sample, after_sample=self.after_sample, after_update=self.after_update)
        x = uni_pc_inst.sample(img, steps=S, skip_type=shared.opts.uni_pc_skip_type, method="multistep", order=shared.opts.uni_pc_order, lower_order_final=shared.opts.uni_pc_lower_order_final)
        return x.to(device), None

def CustomUniPC_model_wrapper(model, noise_schedule, model_type="noise", model_kwargs={}, guidance_scale=1.0, dt_data=None):
    def expand_dims(v, dims):
        return v[(...,) + (None,)*(dims - 1)]
    def get_model_input_time(t_continuous):
        return (t_continuous - 1. / noise_schedule.total_N) * 1000.
    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, None, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
    def model_fn(x, t_continuous, condition, unconditional_condition):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_scale == 1. or unconditional_condition is None:
            return noise_pred_fn(x, t_continuous, cond=condition)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_continuous] * 2)
            if isinstance(condition, dict):
                assert isinstance(unconditional_condition, dict)
                c_in = dict()
                for k in condition:
                    if isinstance(condition[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_condition[k][i],
                            condition[k][i]]) for i in range(len(condition[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_condition[k],
                            condition[k]])
            elif isinstance(condition, list):
                c_in = list()
                assert isinstance(unconditional_condition, list)
                for i in range(len(condition)):
                    c_in.append(torch.cat([unconditional_condition[i], condition[i]]))
            else:
                c_in = torch.cat([unconditional_condition, condition])
            noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
            #return noise_uncond + guidance_scale * (noise - noise_uncond)
            return dt_data.dynthresh(noise, noise_uncond, guidance_scale, None)
    return model_fn
