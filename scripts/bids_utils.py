from pathlib import Path
import json
import pandas as pd

def index_bids_dataset(datapath):
    '''
    Index an arbitrary BIDS dataset and return a data dictionary containing all images, 
    segmentations and pixel sizes.
    '''
    datapath = Path(datapath)

    # init data_dict by reading 'samples.tsv'
    samples = pd.read_csv(datapath / 'samples.tsv', delimiter='\t')
    data_dict = {}
    for i, row in samples.iterrows():
        subject = row['participant_id']
        sample = row['sample_id']
        if subject not in data_dict:
            data_dict[subject] = {}
        data_dict[subject][sample] = {}

    # populate data_dict
    sample_count = 0
    for subject in data_dict.keys():
        samples = data_dict[subject].keys()
        im_path = datapath / subject / 'micr'
        segs_path = datapath / 'derivatives' / 'labels' / subject / 'micr'
        
        images = list(im_path.glob('*.png'))
        axon_segs = list(segs_path.glob('*_seg-axon-*'))
        myelin_segs = list(segs_path.glob('*_seg-myelin-*'))
        for sample in samples:
            for img in images:
                if sample in str(img):
                    data_dict[subject][sample]['image'] = str(img)
                    sample_count += 1
            for axon_seg in axon_segs:
                if sample in str(axon_seg):
                    data_dict[subject][sample]['axon'] = str(axon_seg)
            for myelin_seg in myelin_segs:
                if sample in str(myelin_seg):
                    data_dict[subject][sample]['myelin'] = str(myelin_seg)
        # add pixel_size (assuming isotropic px size)
        json_sidecar = next((datapath / subject / 'micr').glob('*.json'))
        with open(json_sidecar, 'r') as f:
            sidecar = json.load(f)
        data_dict[subject]['px_size'] = sidecar["PixelSize"][0]
    print(f'{sample_count} samples collected.')

    return data_dict

# data loader providing the myelin map (masks), the bboxes (prompts) 
# and the path to the pre-computed image embeddings
def bids_dataloader(data_dict, maps_path, embeddings_path, sub_list):
    '''
    :param data_dict:       contains img, mask and px_size info per sample per subject
    :param maps_path:       paths to myelin maps (instance masks)
    :param embeddings_path  paths to pre-computed image embeddings
    :param sub_list         subjects included
    '''
    subjects = list(data_dict.keys())
    # # we keep the last subject for testing
    # for sub in subjects[:-1]:
    for sub in subjects:
        if sub not in sub_list:
            continue
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            emb_path = embeddings_path / sub / 'micr' / f'{sub}_{sample}_TEM_embedding.pt'
            bboxes = get_sample_bboxes(sub, sample, maps_path)
            myelin_map = get_myelin_map(sub, sample, maps_path)
            yield (emb_path, bboxes, myelin_map)