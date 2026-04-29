import torch
import copy
from tqdm import tqdm
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total

def run(config, model, loaders, logger_best, logger_general):
    PATIENCE = config['training']['patience']
    MID_DELTA = config['training']['min_delta']
    
    
    device = torch.device(config['training']['device'])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs_no_improve = 0
    
    print(f"Starting Default Training for {config['training']['epochs']} epochs...")

    epochs = config['training']['epochs']
    pbar = tqdm(range(1, epochs + 1), desc="Training")
    
    for epoch in pbar:
        train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        scheduler.step()
        val_acc = evaluate(model, loaders['val'], device)
        test_acc = evaluate(model, loaders['test'], device)
        
        #print(f"Epoch {epoch} | Val Acc: {val_acc:.2%} | Test Acc: {test_acc:.2%}")

        pbar.set_postfix({
            "epoch": f"{epoch}",
            "val_acc": f"{val_acc:.3f}",
            "test_acc": f"{test_acc:.3f}"
            })

        logger_general.log({
            "epoch": epoch,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "sparsity": 0.0,
            "params": sum(p.numel() for p in model.parameters())
        })
        
        if val_acc - best_acc > MID_DELTA:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir / f"{config['experiment_name']}_best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs. "
                  f"No improvement in val_acc for {PATIENCE} consecutive epochs.")
            break

    # Final Evaluation on Test Set
    print("Evaluating best model on Test Set...")
    model.load_state_dict(best_wts)
    test_acc = evaluate(model, loaders['test'], device)
    
    logger_best.log({
        "epoch": config['training']['epochs'],
        "val_acc": best_acc,
        "test_acc": test_acc,
        "sparsity": 0.0,
        "params": sum(p.numel() for p in model.parameters())
    })
    print(f"Test Acc: {test_acc:.2%}")