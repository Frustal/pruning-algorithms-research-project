import argparse
import yaml
import sys
import torch
from pathlib import Path
from src.utils import set_seed, CSVLogger
from src.data import get_dataloaders
from src.model import get_model
from src.methods import default, imp, snip

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
    model = get_model(model_name=config['model']['name'],pretrained=config['model']['pretrained'])

    model_config = config.get('model') or {}
    checkpoint = model_config.get('checkpoint')
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            print(f"Loading existing weights from {path}...")
            state_dict = torch.load(path, map_location=torch.device(config['training']['device']))
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {path} not found. Starting from scratch.")
    
    log_dir = Path("output/logs") / config['experiment_name']
    logger_best = CSVLogger(log_dir)
    log_general_name = "train_history.csv"
    logger_general = CSVLogger(log_dir,log_general_name)
    
    # run
    method = config['method']
    if method == 'default':
        default.run(config, model, loaders, logger_best, logger_general)
    elif method == 'imp':
        imp.run(config, model, loaders, logger_best)
    elif method == 'snip':
        snip.run(config, model, loaders, logger_best)
    else:
        print(f"Unknown method: {method}")

if __name__ == "__main__":
    main()