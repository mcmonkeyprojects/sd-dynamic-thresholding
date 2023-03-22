import torch, math

######################### DynThresh Core #########################

class DynThresh:
    def __init__(self, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, power_val, experiment_mode, maxSteps):
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.maxSteps = maxSteps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.experiment_mode = experiment_mode
        self.power_val = power_val

    def interpretScale(self, scale, mode, min):
        scale -= min
        max = self.maxSteps - 1
        if mode == "Constant":
            pass
        elif mode == "Linear Down":
            scale *= 1.0 - (self.step / max)
        elif mode == "Half Cosine Down":
            scale *= math.cos((self.step / max))
        elif mode == "Cosine Down":
            scale *= math.cos((self.step / max) * 1.5707)
        elif mode == "Linear Up":
            scale *= self.step / max
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos((self.step / max))
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos((self.step / max) * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(self.step / max, self.power_val)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(self.step / max, self.power_val)
        scale += min
        return scale

    def dynthresh(self, cond, uncond, cfgScale, weights):
        mimicScale = self.interpretScale(self.mimic_scale, self.mimic_mode, self.mimic_scale_min)
        cfgScale = self.interpretScale(cfgScale, self.cfg_mode, self.cfg_scale_min)
        # uncond shape is (batch, 4, height, width)
        conds_per_batch = cond.shape[0] / uncond.shape[0]
        assert conds_per_batch == int(conds_per_batch), "Expected # of conds per batch to be constant across batches"
        cond_stacked = cond.reshape((-1, int(conds_per_batch)) + uncond.shape[1:])

        ### Normal first part of the CFG Scale logic, basically
        diff = cond_stacked - uncond.unsqueeze(1)
        if weights is not None:
            diff = diff * weights
        relative = diff.sum(1)

        ### Get the normal result for both mimic and normal scale
        mim_target = uncond + relative * mimicScale
        cfg_target = uncond + relative * cfgScale
        ### If we weren't doing mimic scale, we'd just return cfg_target here

        ### Now recenter the values relative to their average rather than absolute, to allow scaling from average
        mim_flattened = mim_target.flatten(2)
        cfg_flattened = cfg_target.flatten(2)
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        ### Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
        mim_max = mim_centered.abs().max(dim=2).values.unsqueeze(2)
        cfg_max = torch.quantile(cfg_centered.abs(), self.threshold_percentile, dim=2).unsqueeze(2)
        actualMax = torch.maximum(cfg_max, mim_max)

        ### Clamp to the max
        cfg_clamped = cfg_centered.clamp(-actualMax, actualMax)
        ### Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
        cfg_renormalized = (cfg_clamped / actualMax) * mim_max

        ### Now add it back onto the averages to get into real scale again and return
        result = cfg_renormalized + cfg_means
        actualRes = result.unflatten(2, mim_target.shape[2:])

        if self.experiment_mode == 1:
            num = actualRes.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    if num[0][0][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][1][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][2][y][x] > 1.5:
                        num[0][2][y][x] *= 0.5
            actualRes = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 2:
            num = actualRes.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    overScale = False
                    for z in range(0, 4):
                        if abs(num[0][z][y][x]) > 1.5:
                            overScale = True
                    if overScale:
                        for z in range(0, 4):
                            num[0][z][y][x] *= 0.7
            actualRes = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 3:
            coefs = torch.tensor([
                #  R       G        B      W
                [0.298,   0.207,  0.208, 0.0], # L1
                [0.187,   0.286,  0.173, 0.0], # L2
                [-0.158,  0.189,  0.264, 0.0], # L3
                [-0.184, -0.271, -0.473, 1.0], # L4
            ], device=uncond.device)
            resRGB = torch.einsum("laxy,ab -> lbxy", actualRes, coefs)
            maxR, maxG, maxB, maxW = resRGB[0][0].max(), resRGB[0][1].max(), resRGB[0][2].max(), resRGB[0][3].max()
            maxRGB = max(maxR, maxG, maxB)
            print(f"test max = r={maxR}, g={maxG}, b={maxB}, w={maxW}, rgb={maxRGB}")
            if self.step / (self.maxSteps - 1) > 0.2:
                if maxRGB < 2.0 and maxW < 3.0:
                    resRGB /= maxRGB / 2.4
            else:
                if maxRGB > 2.4 and maxW > 3.0:
                    resRGB /= maxRGB / 2.4
            actualRes = torch.einsum("laxy,ab -> lbxy", resRGB, coefs.inverse())

        return actualRes
