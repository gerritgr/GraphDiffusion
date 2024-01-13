with open('imports.py', 'r') as file:
    exec(file.read())



class VectorInference(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(VectorInference, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, dataloader=None, noise_to_start=None, steps=None, *args, **kwargs):
        if isinstance(steps, int):
            steps = np.linspace(1, 0, steps)
        elif steps is None:
            steps = np.linspace(1, 0, 100)
        
        if len(steps) < 2 or not np.all(steps[:-1] > steps[1:]) or not (0 <= steps[0] <= 1 and 0 <= steps[-1] <= 1):
            raise ValueError("Steps must be a decreasing sequence in the range [0, 1] and have a length of at least 2.")
        # Ensure either dataloader or noise_to_start is not None
        if dataloader is None and noise_to_start is None:
            raise ValueError("Either dataloader or noise_to_start must be provided.")

        # Get starting point by adding 100% noise to a random data point
        if noise_to_start is None:
            random_data_point = dataloader.dataset[np.random.randint(len(dataloader.dataset))]
            noise_to_start = pipeline.degradation(random_data_point, t=1.0)

        data_t = noise_to_start
        for i, t in enumerate(steps):
            data_0 = pipeline.reconstruction(data_t, t)
            print("data_0", data_0)
            if i >= len(steps) - 1:
                return data_0
            t_next = steps[i+1]
            data_t_next = pipeline.bridge(data_t, data_0, t, t_next)
            data_t = data_t_next

        # Placeholder inference logic
        return data_t