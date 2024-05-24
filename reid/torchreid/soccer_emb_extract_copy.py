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

    for s_id,seq in tqdm(enumerate(seqs,1), total=len(seqs), desc='Processing Seqs'):
        print(seq)
        seq = seq.replace('.txt','')
        print('processing seq ', s_id)

        imgs = sorted(glob.glob(data_path+'/{}/img1/*'.format(seq)))
        detections = np.genfromtxt(data_path+'/{}/det/det.txt'.format(seq),dtype=float, delimiter=',')

        embs = []

        last_frame = int(detections[-1][0])

        for frame_id in tqdm(range(1,last_frame+1), total=last_frame, desc='Processing frame'):

            # if frame_id%100 == 0:
            #     print('processing frame {}'.format(frame_id))

            inds = detections[:,0] == frame_id
            frame_det = detections[inds]
            img = Image.open(imgs[int(frame_id)-1])
            input = None

            frame_emb = []

            for _,_,x,y,w,h,_,_,_,_ in frame_det:

                im = img.crop((x,y,x+w,y+h))
                im = val_transforms(im.convert('RGB')).unsqueeze(0)
                # if input is None:
                #     input = im
                # else:
                #     input = torch.cat([input, im], dim=0)
                features = extractor(im)
                feat = features.cpu().detach().numpy().tolist()
                frame_emb.append(feat)

            # features = extractor(input)
            # feat = features.cpu().detach().numpy()
            # if embs is None:
            #     embs = feat
            # else:
            #     embs = np.concatenate((embs,feat),axis=0)

            embs.append(frame_emb)
        
        embs = np.array(embs, dtype=object)

        print(embs.shape)
        print(last_frame)
        assert len(embs) == last_frame

        np.save(os.path.join(output_dir, '{}.npy'.format(seq)),embs)