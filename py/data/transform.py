import torchvision.transforms as T

"""
Modified from: https://github.com/thuml/Transfer-Learning-Library 

@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com

Copyright (c) 2018 The Python Packaging Authority
"""


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


# OfficeHome
def _office_home_validation():
    transform = ResizeImage(224)
    transform = T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def _office_home_training():
    transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    transform = [transform]
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
    transform.append(T.RandomGrayscale())
    transform.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = T.Compose(transform)
    return transform


OFFICE_HOME_TRAINING = _office_home_training()
OFFICE_HOME_VALIDATION = _office_home_validation()
