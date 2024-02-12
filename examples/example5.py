from graphdiffusion.pipeline import *
import seaborn as sns
from graphdiffusion.plotting import plot_data_as_normal_pdf

with open("../graphdiffusion/imports.py", "r") as file:
    exec(file.read())
import os

if not os.path.exists("pokemon"):
    os.system("git clone https://github.com/gerritgr/pokemon_diffusion && cp -r pokemon_diffusion/pokemon pokemon/ && rm -rf pokemon_diffusion")
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from PIL import Image

IMG_SIZE = 16
set_all_seeds(1234)


class ImageDataset(Dataset):
    """A dataset for images that supports on-the-fly transformations (i.e., augmentations).

    Args:
        imgpaths (str, optional): The path pattern for the images. Default is "pokemon/*png".
    """

    def __init__(self, img_paths="pokemon/*png"):

        # You can/should play around with these.
        self.on_the_fly_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.RandomCrop(int(IMG_SIZE * 0.8)),
                        ]
                    ),
                    p=0.1,
                ),
                transforms.RandomAutocontrast(p=0.1),
                transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                transforms.Lambda(lambda t: torch.clamp(t, min=-1.0, max=1.0)),
            ]
        )

        self.img_list = list()
        for img_path in glob.glob(img_paths):
            # Turn the transparent part in the image to white following
            # https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
            image = Image.open(img_path)
            rgba_image = Image.new("RGBA", image.size, "WHITE")
            rgba_image.paste(image, (0, 0), image)
            rgb_image = rgba_image.convert("RGB")

            # Convert the PIL image to a tensor, where each value is in [-1,1].
            img_as_tensor = image_to_tensor(rgb_image, img_size=IMG_SIZE)
            self.img_list.append(img_as_tensor)

    def __getitem__(self, index):
        """Get an image tensor from the dataset with on-the-fly transformation.

        Args:
            index (int): The index of the image tensor in the dataset.

        Returns:
            torch.Tensor: The image tensor with on-the-fly transformation.
        """
        img = self.img_list[index]
        img = self.on_the_fly_transform(img)

        return img.flatten()

    def get_pil_image(self, index, with_random_augmentation=True):
        """Get a PIL image from the dataset with or without on-the-fly transformation.

        Args:
            index (int): The index of the PIL image in the dataset.
            with_random_augmentation (bool, optional): Whether to apply on-the-fly transformation. Default is True.

        Returns:
            PIL.Image: The PIL image with or without on-the-fly transformation.
        """
        if with_random_augmentation:
            return tensor_to_img(self.__getitem__(index))
        return tensor_to_img(self.img_list[index].reshape(3, IMG_SIZE, IMG_SIZE))

    def __len__(self):
        return len(self.img_list)


dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

dataloader_show = DataLoader(dataset, batch_size=1, shuffle=True)


def plot_image_on_axis(array, axis, arrays=None):
    array = array[0, :]
    array = array.reshape(3, IMG_SIZE, IMG_SIZE)
    array = tensor_to_img(torch.tensor(array))
    array = np.clip(array, 0, 255)  # should be redundant
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = array / 255.0

    # Display the image
    axis.imshow(array.astype(np.uint8) if array.dtype == np.float32 or array.dtype == np.float64 else array)


# pre_trained_path="../pre_trained/vectordenoiser_pokemon_weights.pt"
degradation_obj = VectorDegradationDDPM()
reconstruction_obj = VectorDenoiserLarge(node_feature_dim=3 * IMG_SIZE * IMG_SIZE)
pipeline = PipelineVector(
    node_feature_dim=3 * IMG_SIZE * IMG_SIZE,
    reconstruction_obj=reconstruction_obj,
    degradation_obj=degradation_obj,
    ddpm_end=0.04,
    pre_trained_path="../pre_trained/vectordenoiser_pokemon_weights.pt",
    clamp_inference=(-1.1, 1.1),
    level="DEBUG",
)
pipeline.visualize_foward(
    data=dataloader_show,
    outfile="images/example5/pokemon_forward.jpg",
    plot_data_func=plot_image_on_axis,
    num=25,
)


# pipeline.train(data=dataloader, epochs=10000)
# pipeline.save_all_model_weights("../pre_trained/vectordenoiser_pokemon_weights.pt")


# pipeline.config["vectorbridge_magnitude_scale"] = 1.0
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/pokemonbackward_normal.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeNaive()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/pokemonbackward_naive.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeColdDiffusion()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/pokemon_backward_cold.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeDDPM()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/pokemon_backward_ddpm.jpg",
    num=36,
    steps=100,
)

pipeline.bridge_obj = VectorBridgeAlt()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/pokemon_backward_alt.jpg",
    num=36,
    steps=100,
)

########################
# Unet
########################

degradation_obj = VectorDegradationDDPM()
reconstruction_obj = Unet(
    dim_mults=(
        1,
        2,
        4,
    ),
    dim=IMG_SIZE,
)
pipeline = PipelineVector(
    node_feature_dim=3 * IMG_SIZE * IMG_SIZE,
    hidden_dim=1024,
    num_layers=12,
    dropout_rate=0.3,
    degradation_obj=degradation_obj,
    reconstruction_obj=reconstruction_obj,
    clamp_inference=(-1.1, 1.1),
    pre_trained_path="../pre_trained/unet/pokemon_weights.pt",
    level="DEBUG",
)
pipeline.visualize_foward(
    data=dataloader_show,
    outfile="images/example5/pokemon_forward.jpg",
    plot_data_func=plot_image_on_axis,
    num=25,
)

# print("reconstruction_obj", str(reconstruction_obj))#, [k for k,v in reconstruction_obj.named_parameters()])


pipeline.train(data=dataloader, epochs=100)
pipeline.save_all_model_weights("../pre_trained/unet_pokemon_weights.pt")


pipeline.config["vectorbridge_magnitude_scale"] = 1.0
pipeline.config["vectorbridge_rand_scale"] = 5.0
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/unet/backward_normal.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeNaive()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/unet/backward_naive.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeColdDiffusion()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/unet/backward_cold.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeDDPM()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/unet/backward_ddpm.jpg",
    num=36,
    steps=100,
)

pipeline.bridge_obj = VectorBridgeAlt()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example5/unet/backward_alt.jpg",
    num=36,
    steps=100,
)
