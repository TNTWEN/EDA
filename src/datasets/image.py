from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def pad_shorter_edge_to_minimum(image, minimum_size):
    width, height = image.size
    # 计算当前图像的宽和高与指定的最小尺寸之间的差距
    width_padding = max(0, minimum_size - width)
    height_padding = max(0, minimum_size - height)
    
    # 计算填充的左、右、上、下边距
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding
    
    # 使用Pillow的Pad函数进行填充
    padded_image = transforms.Pad((left_padding, top_padding, right_padding, bottom_padding))(image)
    
    return padded_image



class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root,minimum_size, transform=None, split="train",):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform
        self.minimum_size = minimum_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        img = pad_shorter_edge_to_minimum(img,self.minimum_size)
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)