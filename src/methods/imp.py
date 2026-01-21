import torch
import copy
from pathlib import Path
from tqdm import tqdm
from .default import train_one_epoch, evaluate

def prune_model(model, sparsity, device):
    """Calculates threshold and applies mask to weights."""
    all_weights = []
    # gathering prunable weights (Conv2d, Linear)
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            all_weights.append(m.weight.data.abs().view(-1))
    
    if not all_weights: return {}, 0
    
    # finding global threshold
    flat = torch.cat(all_weights)
    k = int(len(flat) * sparsity)
    threshold = torch.kthvalue(flat, k).values.item()
    
    # creating masks
    masks = {}
    active_params = 0
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            mask = (m.weight.data.abs() > threshold).float().to(device)
            masks[name] = mask
            # applying
            with torch.no_grad():
                m.weight.data.mul_(mask)
            active_params += mask.sum().item()
            
    return masks, active_params

def run(config, model, loaders, logger):
    device = torch.device(config['training']['device'])
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # phase 1: initial training
    print(f"Starting Default Training for {config['training']['epochs']} epochs...")
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9)
    best_base_acc = 0.0
    best_base_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, config['training']['epochs'] + 1):
        train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_acc = evaluate(model, loaders['val'], device)

        print(f"Epoch {epoch} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_base_acc:
            best_base_acc = val_acc
            best_base_wts = copy.deepcopy(model.state_dict())
            
    print(f"Base Training Complete. Best Val: {best_base_acc:.2%}")
    
    dense_model_state = copy.deepcopy(best_base_wts)
    
    for sparsity in config['pruning']['targets']:
        # phase 2: pruning
        print(f"\n--- Pruning to Target: {sparsity:.0%} ---")
        
        model.load_state_dict(dense_model_state)
        
        masks, active_params = prune_model(model, sparsity, device)
        
        # phase 3: retrain
        optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9)
        best_sparse_acc = 0.0
        best_sparse_wts = copy.deepcopy(model.state_dict())
        
        epochs = config['pruning']['retrain_epochs']
        for epoch in range(1, epochs + 1):
            model.train()
            for images, labels in tqdm(loaders['train'], desc="Retraining", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                
                # enforcing masks after update
                with torch.no_grad():
                    for name, m in model.named_modules():
                        if name in masks:
                            m.weight.data.mul_(masks[name])
                            
            # validation
            print(f"Epoch {epoch} | Val Acc: {val_acc:.2%}")
            
            val_acc = evaluate(model, loaders['val'], device)
            if val_acc > best_sparse_acc:
                best_sparse_acc = val_acc
                best_sparse_wts = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_sparse_wts)
        with torch.no_grad():
            for name, m in model.named_modules():
                if name in masks: m.weight.data.mul_(masks[name])

        print("Evaluating best model on Test Set...")
        test_acc = evaluate(model, loaders['test'], device)
        
        logger.log({
            "epoch": epochs,
            "val_acc": best_sparse_acc,
            "test_acc": test_acc,
            "sparsity": sparsity,
            "params": active_params
        })
        torch.save(model.state_dict(), save_dir / f"{config['experiment_name']}_sp{int(sparsity*100)}.pth")
        print(f"Result {sparsity:.0%} | Test Acc: {test_acc:.2%}")