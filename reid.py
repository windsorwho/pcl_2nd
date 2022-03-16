import os
import json
import numpy as np
import torch
from sklearn.preprocessing import normalize
import gc

def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')

class NewDataset(torch.utils.data.Dataset):
    def __init__(self, path_list) -> None:
        super().__init__()
        self.path_list = path_list

    def __getitem__(self, index):
        path = self.path_list[index]
        fea = np.fromfile(path, dtype='<f4')
        feature = torch.from_numpy(np.array(fea))

        return feature

    def __len__(self):
        return len(self.path_list)

@torch.no_grad()
def reid(bytes_rate):
    torch.cuda.empty_cache()
    DEBUG  = False
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}/'.format(bytes_rate)
    gallery_fea_dir = 'gallery_feature/'
    reid_results_path = 'reid_results/{}.json'.format(bytes_rate)

    if DEBUG:
        reconstructed_query_fea_dir = 'project/' + reconstructed_query_fea_dir
        gallery_fea_dir = 'project/' + gallery_fea_dir
        reid_results_path = 'project/' + reid_results_path
        print(gallery_fea_dir, reconstructed_query_fea_dir, reid_results_path)

    os.makedirs(os.path.dirname(reid_results_path), exist_ok=True)

    query_names = os.listdir(reconstructed_query_fea_dir)
    gallery_names = os.listdir(gallery_fea_dir)

    test_query_list = np.array([reconstructed_query_fea_dir + i for i in query_names])
    test_gallery_list = np.array([gallery_fea_dir + i for i in gallery_names])

    if DEBUG:
        print(test_query_list, test_gallery_list)

    batch_size = 32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    query_dataset = NewDataset(test_query_list)
    query_dataloader = torch.utils.data.DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False)

    gallery_dataset = NewDataset(test_gallery_list)
    gallery_dataloader = torch.utils.data.DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False)


    test_query = torch.zeros(len(test_query_list), 2048, device=device)
    test_gallery = torch.zeros(len(test_gallery_list), 2048, device=device)

    for i, data in enumerate(query_dataloader):
        test_query[i*batch_size:i*batch_size+len(data)].copy_(data)

    for i, data in enumerate(gallery_dataloader):
        test_gallery[i*batch_size:i*batch_size+len(data)].copy_(data)


    # test_query = torch.nn.functional.normalize(test_query, p=2, dim=1)
    # test_gallery = torch.nn.functional.normalize(test_gallery, p=2, dim=1)
    # sim = torch.mm(test_query, test_gallery.t())
    # indece = torch.argsort(-sim, dim=1)[:, :1000]

    total_idx = 0
    sub = {}
    for idx in range(test_query.shape[0]//100 + 1):

        idss = torch.mm(test_query[idx*100: (idx+1)*100], test_gallery.t())
        indice = torch.argsort(-idss, dim=1)[:, :100].cpu().numpy()
        for i in range(idss.shape[0]):
            ids_path = test_gallery_list[indice[i]]
            base_name = [os.path.basename(x) for x in ids_path]
            base_name = [x.split('.')[0]+'.png' for x in base_name]

            sub_name = os.path.basename(test_query_list[total_idx])
            sub_name = sub_name.split('.')[0]+'.png'
            sub[sub_name] = base_name
            total_idx += 1

    if total_idx < test_query.shape[0]:
        idss = torch.mm(test_query[total_idx:], test_gallery.t())
        indice = torch.argsort(-idss, dim=1)[:, :100].cpu().numpy()
        for i in range(idss.shape[0]):
            ids_path = test_gallery_list[indice[i]]
            base_name = [os.path.basename(x) for x in ids_path]
            base_name = [x.split('.')[0]+'.png' for x in base_name]

            sub_name = os.path.basename(test_query_list[total_idx])
            sub_name = sub_name.split('.')[0]+'.png'
            sub[sub_name] = base_name
            total_idx += 1

    with open(reid_results_path, 'w', encoding='UTF8') as f:
        f.write(json.dumps(sub, indent=2, sort_keys=False))

    for x in list(locals().keys())[:]:
        del locals()[x]
    gc.collect()

    print('ReID Done')



if __name__ == '__main__':
    reid(64)
