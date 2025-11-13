import os
import argparse
import yaml
import importlib
import logging
import time
import glob
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Rich 라이브러리
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console

# -----------------------------------------------------------------------------
# Utils: AverageMeter & Logger
# -----------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val_dict, n=1):
        for key, value in val_dict.items():
            if key not in self.sum:
                self.sum[key] = 0
                self.count[key] = 0
                self.avg[key] = 0
            
            self.val[key] = value
            self.sum[key] += value * n
            self.count[key] += n
            self.avg[key] = self.sum[key] / self.count[key]

    def get_str(self):
        return " | ".join([f"{k}: {v:.6f}" for k, v in self.avg.items()])

def setup_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러만 설정 (화면 출력 X)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.propagate = False
    
    return logger

def log_to_console(console, msg, logger=None):
    """
    중요한 로그(Validation 결과 등)는 화면에 줄바꿈으로 출력하고 파일에도 기록
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"{timestamp} - {msg}"
    
    if console:
        console.print(formatted_msg)
    else:
        print(formatted_msg)

    if logger:
        logger.info(msg)

# -----------------------------------------------------------------------------
# Custom Dataset & Functions (기존과 동일)
# -----------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mode='train'):
        self.img_dir = img_dir
        self.mode = mode
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        
        if len(self.img_paths) == 0:
            print(f"Warning: No .png files found in {img_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("YCbCr")
            y_image, _, _ = image.split()
        except Exception as e:
            y_image = Image.new('L', (64, 64))

        w, h = y_image.size
        crops = []
        bh, bw = 8, 8

        if self.mode == 'train':
            num_crops = 16
            for _ in range(num_crops):
                if w > bw and h > bh:
                    left = random.randint(0, w - bw)
                    top = random.randint(0, h - bh)
                else:
                    left, top = 0, 0
                crop = y_image.crop((left, top, left + bw, top + bh))
                crops.append(self.to_tensor(crop))
        else:
            centers = [
                (w // 4, h // 4), (w * 3 // 4, h // 4),
                (w // 4, h * 3 // 4), (w * 3 // 4, h * 3 // 4)
            ]
            for cx, cy in centers:
                left = max(0, min(cx - bw // 2, w - bw))
                top = max(0, min(cy - bh // 2, h - bh))
                crop = y_image.crop((left, top, left + bw, top + bh))
                crops.append(self.to_tensor(crop))

        return torch.stack(crops)

def get_model(model_name, **kwargs):
    try:
        module = importlib.import_module(model_name)
        model_class = getattr(module, model_name)
        return model_class(**kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}'. Error: {e}")

def adjust_learning_rate(optimizer, epoch, base_lr, milestones, multipliers):
    lr = base_lr
    for mil, mult in zip(milestones, multipliers):
        if epoch >= mil:
            lr = base_lr * mult
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def validate(model, val_loader, device, epoch, logger, console):
    model.eval()
    meters = AverageMeter()
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            b, n, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w) 
            targets = inputs 
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss_dict = {'val_loss': loss.item()}
            meters.update(loss_dict, inputs.size(0))

    msg = f"\n[Validation Epoch {epoch}] Result: {meters.get_str()}"
    log_to_console(console, msg, logger)
    log_to_console(console, "-" * 80 + "\n", logger)
    
    return meters.avg['val_loss']

# -----------------------------------------------------------------------------
# Main Train Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger(cfg['log_dir'])
    console = Console()
    
    log_to_console(console, "Loaded Configuration", logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = CustomImageDataset(cfg['train_dataset_path'], mode='train')
    val_dataset = CustomImageDataset(cfg['val_dataset_path'], mode='val')

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    log_to_console(console, f"Loading Model: {cfg['model_name']}", logger)
    model = get_model(cfg['model_name'], **cfg['model_params']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['base_lr'])
    criterion = nn.MSELoss()

    start_epoch = 0

    if args.resume:
        ckpt_path = args.checkpoint
        if ckpt_path and os.path.isfile(ckpt_path):
            log_to_console(console, f"Resuming from: {ckpt_path}", logger)
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            log_to_console(console, "Checkpoint not found. Starting from scratch.", logger)

    total_epochs = cfg['epochs']
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    current_global_step = start_epoch * steps_per_epoch

    # [핵심 수정 1] Progress 정의에 TextColumn("{task.fields[info]}") 추가
    # style="bold yellow" 등으로 색상 지정 가능
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold yellow]{task.fields[info]}"), # <-- 여기에 동적 정보 표시
        console=console
    )

    log_to_console(console, "Running initial validation...", logger)
    validate(model, val_loader, device, start_epoch - 1, logger, console)

    refresh_freq = 10
    with progress:
        # [핵심 수정 2] task 생성 시 info 필드 초기화
        task_id = progress.add_task("[green]Training...", total=total_steps, completed=current_global_step, info="")

        for epoch in range(start_epoch, total_epochs):
            model.train()
            meters = AverageMeter()
            
            curr_lr = adjust_learning_rate(optimizer, epoch, cfg['base_lr'], cfg['lr_milestones'], cfg['lr_multipliers'])
            
            if epoch == start_epoch:
                log_to_console(console, f"Start Epoch {epoch}/{total_epochs} | LR: {curr_lr:.6f}", logger)

            for i, inputs in enumerate(train_loader):
                inputs = inputs.to(device)
                b, n, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)
                targets = inputs 

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metric update
                logs_dict = {'loss': loss.item()}
                meters.update(logs_dict)

                # [최적화된 로직]
                # 1. 기본적으로 진행 바(Bar)는 매번 1칸씩 전진시킵니다.
                advance_steps = 1

                # 2. 텍스트 정보(info)는 특정 주기마다만 계산하고 업데이트합니다.
                if (i + 1) % refresh_freq == 0:
                    info_str = f"Loss: {meters.avg['loss']:.6f} | LR: {curr_lr:.6f}"
                    progress.update(task_id, advance=advance_steps, info=info_str)
                else:
                    # 텍스트 갱신 없이 바만 전진 (훨씬 가벼움)
                    progress.update(task_id, advance=advance_steps)

                # 3. 파일 로그 저장 (기존 유지)
                if (i + 1) % cfg['print_freq'] == 0:
                    # 파일에 쓸 때는 가장 최신 info_str을 다시 만들거나 위 변수 재사용
                    log_msg = f"Epoch [{epoch}/{total_epochs}] Step [{i+1}/{steps_per_epoch}] Loss: {meters.avg['loss']:.6f} | LR: {curr_lr:.6f}"
                    logger.info(log_msg)

            # Epoch 끝날 때 Validation
            validate(model, val_loader, device, epoch, logger, console)

            save_path = os.path.join(cfg['save_dir'], f"checkpoint_epoch_{epoch}.pth")
            Path(cfg['save_dir']).mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': meters.avg['loss'],
            }, save_path)

    log_to_console(console, "Training Finished.", logger)

if __name__ == "__main__":
    main()
