import argparse
import yaml
import sys
from pathlib import Path
from src.utils import set_seed, CSVLogger
from src.data import get_dataloaders
from src.model import get_model
from src.methods import default, imp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.debug:
        print("DEBUG MODE: 1 Epoch, Small Data Subset")
        config['training']['epochs'] = 1
        if 'pruning' in config: config['pruning']['retrain_epochs'] = 1
        config['experiment_name'] += "_DEBUG"

    set_seed(config['seed'])
    
    # data
    loaders = get_dataloaders(config['data']['batch_size'], config['data']['num_workers'])
    
    # debug data subset
    if args.debug:
        import itertools
        for k in loaders:
            loaders[k] = list(itertools.islice(loaders[k], 2)) # 2 batches only

    # model and logger
    model = get_model(pretrained=config['model']['pretrained'])

    checkpoint = config.get('model', {}).get('checkpoint')
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            print(f"Loading existing weights from {path}...")
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {path} not found. Starting from scratch.")
    
    log_dir = Path("output/logs") / config['experiment_name']
    logger = CSVLogger(log_dir)

    # run
    method = config['method']
    if method == 'default':
        default.run(config, model, loaders, logger)
    elif method == 'imp':
        imp.run(config, model, loaders, logger)
    else:
        print(f"Unknown method: {method}")

if __name__ == "__main__":
    main()