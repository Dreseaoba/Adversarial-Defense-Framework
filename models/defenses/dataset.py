import os
from PIL import Image
from torch.utils.data import Dataset



class AdvDataset(Dataset):
    def __init__(
        self,
        ori:str,
        adv:str,
        transform
    ) -> None:
        self.ori = ori
        self.adv = adv
        self.ori_imgs = os.listdir(ori)
        self.adv_imgs = os.listdir(adv)
        if len(self.ori_imgs) != len(self.adv_imgs):
            raise NotImplementedError('ori and adv not same num')
        for i in range(len(self.ori_imgs)):
            if self.ori_imgs[i] != self.adv_imgs[i]:
                raise NotImplementedError(self.ori_imgs[i], self.adv_imgs[i])
        self.tranform = transform

    def pil_loader(self, path: str):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index: int):
        ori = self.tranform(self.pil_loader(os.path.join(self.ori, self.ori_imgs[index])))
        adv = self.tranform(self.pil_loader(os.path.join(self.adv, self.adv_imgs[index])))
        return adv, ori


    def __len__(self) -> int:
        return len(self.ori_imgs)