import os
import os.path as osp
#from torch.backends import cudnn
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
#from config import Config
from scipy.spatial import distance
import glob
import sys
from utils import FeatureExtractor
import torchreid
from tqdm import tqdm


if __name__ == "__main__":
    #cfg = Config()
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = '../checkpoints/sports_model.pth.tar-60',
        device='cuda'
    ) 
    
    data_path = '/home/psc/Desktop/SoccerNet/tracking/test'

    output_dir = os.path.join('/home/psc/Desktop/SoccerNet/embs1', os.path.basename(data_path))
    os.makedirs(output_dir, exist_ok=True)

    seqs = sorted(os.listdir(data_path))

    for s_id,seq in tqdm(enumerate(seqs), total=len(seqs), desc='Processing Sequence'):
        print(seq)
        seq = seq.replace('.txt','')
        print('processing seq ', s_id)

        imgs = sorted(glob.glob(data_path+'/{}/img1/*'.format(seq)))
        detections = np.genfromtxt(data_path+'/{}/det/det.txt'.format(seq),dtype=float, delimiter=',')

        embs = []

        # last_frame = int(detections[-1][0])

        for frame_id, dets in tqdm(enumerate(detections)):
            print(type(dets), dets)

            # inds = detections[:,0] == frame_id
            # frame_det = detections[inds]
            img = Image.open(imgs[int(frame_id)-1])
            frame_emb = []

            for _,_,x,y,w,h,_,_,_,_ in dets:

                im = img.crop((x,y,x+w,y+h))
                im = val_transforms(im.convert('RGB')).unsqueeze(0)
                # if input is None:
                #     input = im
                # else:
                #     input = torch.cat([input, im], dim=0)
                features = extractor(im)
                feat = features.cpu().detach().numpy().tolist()
                frame_emb.append(feat)

            embs.append(frame_emb)

        embs = np.array(embs)

        assert len(embs) == len(detections)

        np.save(os.path.join(output_dir, '{}.npy'.format(seq)),embs)