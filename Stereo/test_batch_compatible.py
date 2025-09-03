from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from models import *
from PIL import Image
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet batch test')
parser.add_argument('--loadmodel', required=True, help='loading model')
parser.add_argument('--left_folder', required=True, help='folder with left images')
parser.add_argument('--right_folder', required=True, help='folder with right images')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load model
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    raise ValueError("No model selected")

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# transform
normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
infer_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(**normal_mean_var)])

# create output folder
output_folder = Path("pre_disp_compitable")
output_folder.mkdir(exist_ok=True, parents=True)

# list images
left_images = sorted(Path(args.left_folder).glob("*.png"))
right_images = sorted(Path(args.right_folder).glob("*.png"))

def test(imgL, imgR):
    model.eval()
    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()
    with torch.no_grad():
        disp = model(imgL, imgR)
    return torch.squeeze(disp).cpu().numpy()

for left_img, right_img in tqdm(zip(left_images, right_images), total=len(left_images)):
    imgL_o = Image.open(left_img).convert('RGB')
    imgR_o = Image.open(right_img).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)

    # pad to 16 times
    top_pad = (16 - imgL.shape[1] % 16) % 16
    right_pad = (16 - imgL.shape[2] % 16) % 16
    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL, imgR)
    print(f"{left_img.name} processed in {time.time() - start_time:.2f}s")

    # remove padding
    if top_pad !=0 and right_pad !=0:
        pred_disp = pred_disp[top_pad:, :-right_pad]
    elif top_pad ==0 and right_pad !=0:
        pred_disp = pred_disp[:, :-right_pad]
    elif top_pad !=0 and right_pad ==0:
        pred_disp = pred_disp[top_pad:, :]
    
    # resize to GT shape (1200x1920)
    pred_disp = np.array(Image.fromarray(pred_disp).resize((1920, 1200), resample=Image.BILINEAR))

    # scale to GT range and convert to uint8
    max_val = pred_disp.max() if pred_disp.max() > 0 else 1
    scale_factor = 170 / max_val
    pred_disp = (pred_disp * scale_factor).astype('uint8')

    # save
    Image.fromarray(pred_disp).save(output_folder / left_img.name)

print("All disparities saved in 'pre_disp' folder.")
