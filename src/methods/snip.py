import torch
import copy
from pathlib import Path
from tqdm import tqdm
from .default import evaluate

def prune_model_snip(model, loader, criterion, sparsity, device):
    """Calculates SNIP sensitivities and applies mask to weights."""
    model.eval()
    
    # Grab one batch
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    model.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    
    all_sensitivities = []
    # gathering prunable weights (Conv2d, Linear)
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if m.weight.grad is not None:
                sensitivity = (m.weight.data * m.weight.grad).abs().view(-1)
                all_sensitivities.append(sensitivity)
    
    if not all_sensitivities: return {}, 0
    
    # finding global threshold
    flat = torch.cat(all_sensitivities)
    k = int(len(flat) * sparsity)
    threshold = torch.kthvalue(flat, k).values.item()
    
    # creating masks
    masks = {}
    active_params = 0
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if m.weight.grad is not None:
                sensitivity = (m.weight.data * m.weight.grad).abs()
                mask = (sensitivity > threshold).float().to(device)
                masks[name] = mask
                # applying
                with torch.no_grad():
                    m.weight.data.mul_(mask)
                active_params += mask.sum().item()
            
    model.zero_grad()
    return masks, active_params

def run(config, model, loaders, logger):
    device = torch.device(config['training']['device'])
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # SNIP is a single-shot pruning method applied before training
    init_state = copy.deepcopy(model.state_dict())
    
    for sparsity in config['pruning']['targets']:
        print(f"\n--- SNIP Pruning to Target: {sparsity:.0%} ---")
        
        # Reset to initial weights for each sparsity target
        model.load_state_dict(init_state)
        
        masks, active_params = prune_model_snip(model, loaders['train'], criterion, sparsity, device)
        
        # Training the pruned model
        print(f"Training pruned model for {config['training']['epochs']} epochs...")
        optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9, weight_decay=1e-4)
        best_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())
        
        epochs = config['training']['epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(1, epochs + 1):
            model.train()
            for images, labels in tqdm(loaders['train'], desc="Training", leave=False):
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
                            
            scheduler.step()
            # validation
            val_acc = evaluate(model, loaders['val'], device)
            print(f"Epoch {epoch} | Val Acc: {val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_wts)
        with torch.no_grad():
            for name, m in model.named_modules():
                if name in masks: m.weight.data.mul_(masks[name])

        print("Evaluating best model on Test Set...")
        test_acc = evaluate(model, loaders['test'], device)
        
        logger.log({
            "epoch": epochs,
            "val_acc": best_acc,
            "test_acc": test_acc,
            "sparsity": sparsity,
            "params": active_params
        })
        torch.save(model.state_dict(), save_dir / f"{config['experiment_name']}_sp{int(sparsity*100)}.pth")
        print(f"Result {sparsity:.0%} | Test Acc: {test_acc:.2%}")
