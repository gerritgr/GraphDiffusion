import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

from degradation import *


# Define a simple default bridge class
class VectorBridgeNaive(nn.Module):
    def __init__(self, pipeline=None):
        super(VectorBridge, self).__init__()

    def forward(
        self, data_now, data_prediction, t_now, t_query, *args, **kwargs
    ):
        return pipeline.degradation(
            data_prediction, t_query
        )  # this is not actually a bridge distribution between the two points


class VectorBridgeColdDiffusion(nn.Module):
    def __init__(self, pipeline=None):
        super(VectorBridgeColdDiffusion, self).__init__()

    def forward(
        self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs
    ):
        x_0 = data_prediction
        seed = torch.randint(0, 10000, (1,)).item()
        assert t_now > t_query
        x_tminus1 = (
            data_now
            - pipeline.degradation(x_0, t_now, seed=seed)
            + pipeline.degradation(x_0, t_query, seed=seed)
        )
        return x_tminus1


class VectorBridge(nn.Module):
    def __init__(self, pipeline=None):
        super(VectorBridge, self).__init__()

    def forward(
        self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs
    ):
        direction = data_prediction - data_now
        direction = direction / torch.norm(direction)
        seed = torch.randint(0, 10000, (1,)).item()
        magnitude = torch.norm(
            pipeline.degradation(data_prediction, t_now, seed=seed)
            - pipeline.degradation(data_prediction, t_query, seed=seed)
        )

        x_tminus1 = data_now + direction * magnitude  # (1.0-t_now)

        if t_query > 1e-3:
            x_tminus1 = x_tminus1 + torch.randn_like(x_tminus1) * magnitude / 3.0

        return x_tminus1


class VectorBridgeLinear(nn.Module):
    def __init__(self, pipeline=None):
        super(VectorBridgeLinear, self).__init__()

    def forward(
        self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs
    ):
        direction = data_prediction - data_now
        direction = direction / torch.norm(direction)
        x_tminus1 = data_now + direction * (1.0 - t_now)

        if t_query > 1e-3:
            x_tminus1 = x_tminus1 + torch.randn_like(x_tminus1) * t_query / 2.0

        return x_tminus1


class VectorBridgeDDPM(nn.Module):
    def __init__(self, pipeline=None):
        super(VectorBridgeDDPM, self).__init__()

    @staticmethod
    def get_noise_from_pred(x_0_pred, x_t, alphas_cumprod_t):
        # Solve Algorithm 1 from the DDPM paper solved for nosie
        noise = x_t - torch.sqrt(alphas_cumprod_t) * x_0_pred
        noise = noise / torch.sqrt(1.0 - alphas_cumprod_t)
        return noise

    def forward(
        self, data_now, data_prediction, t_now, t_query, pipeline, *args, **kwargs
    ):
        """
        Performs one step of denoising using the provided model.
        Use only with VectorDegradationDDPM.
        We use assume t_query is one step ahead of t_now.
        """
        row_num = data_now.shape[0]
        assert row_num == data_prediction.shape[0]

        # Generate and calculate betas, alphas, and related parameters
        betas = VectorDegradationDDPM.generate_schedule(step_num=pipeline.step_num)
        step_num = betas.numel()
        t_int = int(t_now * (step_num - 1))
        t_query_int = int(t_query * (step_num - 1))
        # print("t_int", t_int, "t_query_int", t_query_int)
        assert (
            t_int == t_query_int + 1 or t_int == t_query_int + 2 or t_int == t_query_int
        )
        t_step_tensor = torch.full((row_num, 1), t_int, device=data_now.device).long()

        beta_t = betas[t_int]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_t = alphas_cumprod[t_int]

        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(alphas[t_int])
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        noise_prediction = VectorBridgeDDPM.get_noise_from_pred(
            data_prediction, data_now, alphas_cumprod_t
        )
        # Compute denoised values
        model_mean = sqrt_recip_alphas_t * (
            data_now - beta_t / sqrt_one_minus_alphas_cumprod_t * noise_prediction
        )
        values_one_step_denoised = model_mean  # in case that t == 0

        if t_int != 0:
            posterior_variance = (
                (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
            )  # in the paper this is in 3.2. Note that sigma^2 is variance, not std.
            posterior_std_t = torch.sqrt(posterior_variance[t_int])
            # posterior_std_t = torch.sqrt(beta_t) #alternative
            noise = rand_like_with_seed(data_now)
            values_one_step_denoised = model_mean + posterior_std_t * noise

        if (
            torch.isnan(values_one_step_denoised).any()
            or torch.isinf(values_one_step_denoised).any()
        ):
            warn_str = "Bridge: The tensor contains NaN or Inf values. These will be replaced with 0."
            print(warn_str)
            print("model_mean", model_mean)
            print("noise_prediction", noise_prediction)
            print("data_prediction", data_prediction)
            print("data_now", data_now)
            print("alphas_cumprod_t", alphas_cumprod_t)
            values_one_step_denoised = torch.where(
                torch.isnan(values_one_step_denoised)
                | torch.isinf(values_one_step_denoised),
                torch.zeros_like(values_one_step_denoised),
                values_one_step_denoised,
            )

        return values_one_step_denoised
