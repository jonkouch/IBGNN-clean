import os
from helpers.constants import ROOT_DIR



def exp_path(dataset_name: str, num_communities, encode_dim, epochs, lr, num_negative, normalize,
             load_epoch: int = 0) -> str:

    epochs = epochs + load_epoch
    run_folder = f'{dataset_name}/' \
                f'com{num_communities}_' \
                f'Enc{int(encode_dim)}_' \
                f'ep{epochs}_' \
                f'lf{lr}_' \
                f'neg{num_negative}_' \
                f'norm{normalize}_'


                 
    return os.path.join(ROOT_DIR, 'pykeen_results', run_folder)