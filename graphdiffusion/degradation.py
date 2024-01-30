import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())

from utils import *

# Define a simple default forward class
class VectorDegradation(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorDegradation, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data, t, seed = None, *args, **kwargs):
        if torch.is_tensor(t):
            batch_dim = data.shape[0]
            t = t.reshape(batch_dim,1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
        else:
            raise TypeError("'t' must be a float or a tensor.")

        if isinstance(t, float) and t < 1e-7:
            return data

        # If t is a tensor, this operation will be applied element-wise.
        t = t**3.0  # Add scaling
        mean = data * (1 - t)
        std_dev = t**0.5 # you could also just ust t

        if seed:
            seed_old = torch.randint(0, 10000, (1,)).item()
            set_all_seeds(seed=seed)

        standard_normal_sample = torch.randn_like(data)

        if seed:
            set_all_seeds(seed=seed_old)


        transformed_sample = mean + std_dev * standard_normal_sample

        return transformed_sample
    



class VectorDegradationDDPM(nn.Module):
    def __init__(self, node_feature_dim=1, step_num = 1000):
        super(VectorDegradationDDPM, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.step_num = step_num
    

    @staticmethod
    def generate_schedule(start = 0.0001, end = 0.01, timesteps=1000):
        """
        Generates a schedule of beta and alpha values for a forward process.

        Args:
        start (float): The starting value for the beta values. Default is START.
        end (float): The ending value for the beta values. Default is END.
        timesteps (int): The number of timesteps to generate. Default is TIMESTEPS.

        Returns:
        tuple: A tuple of three tensors containing the beta values, alpha values, and
        cumulative alpha values (alpha bars).
        """

        betas = torch.linspace(start, end, timesteps)
        assert(betas.numel() == timesteps)
        return betas

    def forward(self, pipeline, data, t, seed = None, *args, **kwargs):
        row_num = data.shape[0]
        feature_dim = data.shape[1]

        if torch.is_tensor(t):
            t = t.reshape(row_num,1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
            t = torch.full((row_num, 1), t, device=data.device)  
        else:
            raise TypeError("'t' must be a float or a tensor.")
    

        t_scaled = torch.round(t * (self.step_num - 1.)).long().to(data.device)
        betas = VectorDegradationDDPM.generate_schedule(timesteps=self.step_num).to(data.device)
        noise = rand_like_with_seed(data, seed=seed)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphabar_t = torch.gather(alphas_cumprod, 0, t_scaled.flatten()).view(row_num, 1)
        assert(alphabar_t.numel() == row_num)

        data_noise_mean = torch.sqrt(alphabar_t) * data # Column-wise multiplication, now it is a matrix
        data_noise_std = torch.sqrt(1.-alphabar_t) # This is a col. vector
        data_noise_std = data_noise_std.repeat(1,feature_dim) # This is a matrix
        data_noise =  data_noise_mean + data_noise_std * noise  

        return data_noise