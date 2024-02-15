import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

from .degradation import *


# Define a simple default bridge class
class VectorBridgeNaive(nn.Module):
    def __init__(self):
        super(VectorBridgeNaive, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs):
        result = pipeline.degradation(data_prediction, t_query)  # this is not actually a bridge distribution between the two points
        return result


class VectorBridgeColdDiffusion(nn.Module):
    def __init__(self):
        super(VectorBridgeColdDiffusion, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs):
        x_0 = data_prediction
        seed = torch.randint(0, 10000, (1,)).item()
        assert t_now > t_query
        x_tminus1 = data_now - pipeline.degradation(x_0, t_now, seed=seed) + pipeline.degradation(x_0, t_query, seed=seed)

        return x_tminus1


class VectorBridgeBackup(nn.Module):
    def __init__(self):
        super(VectorBridgeBackup, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs):
        direction = data_prediction - data_now
        direction = direction / torch.norm(direction)
        seed = torch.randint(0, 10000, (1,)).item()
        magnitude = torch.norm(pipeline.degradation(data_prediction, t_now, seed=seed) - pipeline.degradation(data_prediction, t_query, seed=seed))

        x_tminus1 = data_now + direction * magnitude  # *1.5  # (1.0-t_now)

        if t_query > 1e-3:
            x_tminus1 = x_tminus1 + torch.randn_like(x_tminus1) * magnitude / 3.0

        return x_tminus1


class VectorBridge(nn.Module):
    def __init__(self):
        super(VectorBridge, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, vectorbridge_magnitude_scale=None, vectorbridge_rand_scale=None):
        vectorbridge_magnitude_scale = (
            pipeline.config.vectorbridge_magnitude_scale or 1.0
        )  # vectorbridge_magnitude_scale should be taken automatically from the config, this does not work currently somehow
        vectorbridge_rand_scale = pipeline.config.vectorbridge_rand_scale or 1.0

        vectorbridge_magnitude_scale = 1.0
        vectorbridge_rand_scale = 0.8

        row_num = data_now.shape[0]
        data_orig_shape = data_now.shape
        data_orig_shape = data_prediction.shape
        assert data_orig_shape == data_orig_shape
        data_now = data_now.view(row_num, -1)
        data_prediction = data_prediction.view(row_num, -1)

        direction = data_prediction - data_now
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8  # Prevent division by zero
        direction = direction / direction_norm

        assert direction.shape[0] == row_num
        direction_norm = torch.norm(direction, p=2, dim=1)

        if not torch.allclose(direction_norm.flatten(), torch.ones(row_num).flatten(), atol=1e-3):
            print(direction_norm.flatten())
            print("direction", direction)
            pipeline.warning("The Euclidean lengths are not close to 1")

        seed = torch.randint(0, 10000, (1,)).item()
        degradation_now = pipeline.degradation(data_prediction.view(data_orig_shape), t_now, seed=seed).view(row_num, -1)
        degradation_query = pipeline.degradation(data_prediction.view(data_orig_shape), t_query, seed=seed).view(row_num, -1)
        magnitude = torch.norm(degradation_now - degradation_query, dim=1)
        magnitude = magnitude.view(row_num, 1)

        assert direction.shape[0] == row_num
        assert magnitude.numel() == row_num
        assert (direction * magnitude).shape == direction.shape

        change_vector = direction * magnitude  # multiply each column by the corresponding magnitude
        assert torch.allclose(torch.norm(change_vector, p=2, dim=1).flatten(), magnitude.flatten(), atol=1e-4), "The Euclidean lengths are not close to the given magnitude"

        x_tminus1 = data_now + change_vector / vectorbridge_magnitude_scale

        # Add random noise if t_query is greater than a small threshold to simulate uncertainty
        if t_query > 1e-3:
            x_tminus1 += torch.randn_like(x_tminus1) * magnitude / vectorbridge_rand_scale

        x_tminus1 = x_tminus1.reshape(data_orig_shape)  # Use view_as for safety and readability

        return x_tminus1


class VectorBridgeAlt(nn.Module):
    def __init__(self):
        super(VectorBridgeAlt, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, vectorbridge_magnitude_scale=None, vectorbridge_rand_scale=None):
        vectorbridge_magnitude_scale = (
            pipeline.config.vectorbridge_magnitude_scale or 1.0
        )  # vectorbridge_magnitude_scale should be taken automatically from the config, this does not work currently somehow
        vectorbridge_rand_scale = pipeline.config.vectorbridge_rand_scale or 1.0

        vectorbridge_magnitude_scale = 1.0
        vectorbridge_rand_scale = 2.0

        row_num = data_now.shape[0]
        data_orig_shape = data_now.shape
        data_pred_shape = data_prediction.shape
        assert data_orig_shape == data_pred_shape
        data_now = data_now.view(row_num, -1)
        data_prediction = data_prediction.view(row_num, -1)

        direction = data_prediction - data_now
        # direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8  # Prevent division by zero
        # direction = direction / direction_norm

        assert direction.shape[0] == row_num
        # direction_norm = torch.norm(direction, p=2, dim=1)

        # if not torch.allclose(direction_norm.flatten(), torch.ones(row_num).flatten(), atol=1e-3):
        #    print(direction_norm.flatten())
        #    print("direction", direction)
        #    pipeline.warning("The Euclidean lengths are not close to 1")

        #seed = torch.randint(0, 10000, (1,)).item()
        # degradation_now = pipeline.degradation(data_prediction, t_now, seed=seed).view(row_num, -1)
        # degradation_query = pipeline.degradation(data_prediction, t_query, seed=seed).view(row_num, -1)
        # magnitude = torch.norm(degradation_now - degradation_query, dim=1)
        # magnitude = magnitude.view(row_num, 1)

        assert direction.shape[0] == row_num
        # assert magnitude.numel() == row_num
        # assert (direction * magnitude).shape == direction.shape

        t_diff = t_now - t_query
        assert t_diff >= 0
        assert t_diff <= 1
        assert t_now >= t_diff
        t_scale = t_diff / t_now
        # print(t_scale)
        assert 1 >= t_scale >= 0

        change_vector = direction * 1.1 * t_scale  # (1.0 - t_query) #t_scale # multiply each column by the corresponding magnitude
        # assert torch.allclose(torch.norm(change_vector, p=2, dim=1).flatten(), magnitude.flatten(), atol=1e-4), "The Euclidean lengths are not close to the given magnitude"

        x_tminus1 = data_now + change_vector  # / vectorbridge_magnitude_scale

        # Add random noise if t_query is greater than a small threshold to simulate uncertainty
        if t_query > 1e-3:
            x_tminus1 += torch.randn_like(x_tminus1) * (t_query / vectorbridge_rand_scale)

        x_tminus1 = x_tminus1.reshape(data_orig_shape)  # Use view_as for safety and readability
        return x_tminus1

    # def forward(self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs):
    #    direction = data_prediction - data_now
    #    direction = direction / torch.norm(direction)
    #    x_tminus1 = data_now + direction * (1.0 - t_query)##
    #    if t_query > 1e-3:
    #        x_tminus1 = x_tminus1 + torch.randn_like(x_tminus1) * t_query / 3.0

    #    return x_tminus1


class VectorBridgeDDPM(nn.Module):
    def __init__(self):
        super(VectorBridgeDDPM, self).__init__()

    @staticmethod
    def get_noise_from_pred(x_0_pred, x_t, alphas_cumprod_t):
        # Solve Algorithm 1 from the DDPM paper solved for nosie
        noise = x_t - torch.sqrt(alphas_cumprod_t) * x_0_pred
        noise = noise / torch.sqrt(1.0 - alphas_cumprod_t)
        return noise

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline):
        """
        Performs one step of denoising using the provided model.
        Use only with VectorDegradationDDPM.
        We use assume t_query is one step ahead of t_now.
        """
        row_num = data_now.shape[0]
        assert row_num == data_prediction.shape[0]
        assert t_query <= t_now
        config = pipeline.config

        data_orig_shape = data_now.shape
        data_pred_shape = data_prediction.shape
        assert data_orig_shape == data_pred_shape
        data_now = data_now.view(row_num, -1)
        data_prediction = data_prediction.view(row_num, -1)

        # Generate and calculate betas, alphas, and related parameters
        betas = VectorDegradationDDPM.generate_schedule(step_num=config.step_num, ddpm_start=config.ddpm_start, ddpm_end=config.ddpm_end).to(data_now.device)
        step_num = betas.numel()
        t_int = int(t_now * (step_num - 1))
        t_query_int = int(t_query * (step_num - 1))
        # print("t_int", t_int, "t_query_int", t_query_int)
        assert t_int == t_query_int + 1 or t_int == t_query_int + 2 or t_int == t_query_int
        t_step_tensor = torch.full((row_num, 1), t_int, device=data_now.device).long()

        beta_t = betas[t_int]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_t = alphas_cumprod[t_int]

        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(alphas[t_int])
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        noise_prediction = VectorBridgeDDPM.get_noise_from_pred(data_prediction, data_now, alphas_cumprod_t) # TODO check dimensions
        # Compute denoised values
        model_mean = sqrt_recip_alphas_t * (data_now - beta_t / sqrt_one_minus_alphas_cumprod_t * noise_prediction)
        values_one_step_denoised = model_mean  # in case that t == 0

        if t_int != 0:
            posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas  # in the paper this is in 3.2. Note that sigma^2 is variance, not std.
            if pipeline.config.vectorbridgeddpm_use_simple_posterior_std:
                posterior_std_t = torch.sqrt(beta_t)
            else:
                posterior_std_t = torch.sqrt(posterior_variance[t_int])
            noise = rand_like_with_seed(data_now)

            assert posterior_std_t.shape == noise.shape or posterior_std_t.numel() == 1  # we are in the second case because all time points are the same
            values_one_step_denoised = model_mean + posterior_std_t * noise

        if torch.isnan(values_one_step_denoised).any() or torch.isinf(values_one_step_denoised).any():
            warn_str = "Bridge: The tensor contains NaN or Inf values. These will be replaced with 0."
            pipeline.warn(warn_str)
            pipeline.debug("model_mean", model_mean)
            pipeline.debug("noise_prediction", noise_prediction)
            pipeline.debug("data_prediction", data_prediction)
            pipeline.debug("data_now", data_now)
            pipeline.debug("alphas_cumprod_t", alphas_cumprod_t)
            values_one_step_denoised = torch.where(
                torch.isnan(values_one_step_denoised) | torch.isinf(values_one_step_denoised),
                torch.zeros_like(values_one_step_denoised),
                values_one_step_denoised,
            )

        values_one_step_denoised = values_one_step_denoised.view(*data_orig_shape)
        return values_one_step_denoised
