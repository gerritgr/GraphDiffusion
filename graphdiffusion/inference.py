import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())



class VectorInference(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorInference, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, dataloader=None, noise_to_start=None, steps=None, *args, **kwargs):
        if isinstance(steps, int):
            steps = np.linspace(1, 0, steps)
        elif steps is None:
            steps = np.linspace(1, 0, 100)
        
        if len(steps) < 2 or not np.all(steps[:-1] > steps[1:]) or not (0 <= steps[0] <= 1 and 0 <= steps[-1] <= 1):
            raise ValueError("Steps must be a decreasing sequence in the range [0, 1] and have a length of at least 2.", "steps:", steps)
        # Ensure either dataloader or noise_to_start is not None
        if dataloader is None and noise_to_start is None:
            raise ValueError("Either dataloader or noise_to_start must be provided.")
        
        model = pipeline.get_model()
        model.eval()

        # Get starting point by adding 100% noise to a random data point
        if noise_to_start is None:
            random_data_point = next(iter(dataloader))
            noise_to_start = pipeline.degradation(random_data_point, t=1.0)

        data_t = noise_to_start
        for i, t in enumerate(steps):
        #for i, t in tqdm(enumerate(steps), total=len(steps), desc='Processing inference'):
            data_0 = pipeline.reconstruction(data_t, t)
            if i >= len(steps) - 1:
                if t<1e-3:
                    return data_0, data_0
                else:
                    return data_t, data_0 #pipeline.bridge(data_t, data_0, t, t) # or just return data_t
            t_next = steps[i+1]
            assert(t_next < t)
            data_t_next = pipeline.bridge(data_t, data_0, t, t_next)
            data_t = data_t_next

            if torch.isnan(data_t).any() or torch.isinf(data_t).any():
                warn_str = 'The tensor contains NaN or Inf values. These will be replaced with 0. step:'+str(i)
                print(warn_str)
                data_t = torch.where(torch.isnan(data_t) | torch.isinf(data_t), torch.zeros_like(data_t), data_t)


        return data_t, data_0