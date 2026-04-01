import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def plot_loss_curve(train_loss, val_loss, save_path='loss_curve.png'):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_loss)+1), train_loss, 'b-', linewidth=2, label='Train Loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, 'r-', linewidth=2, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(alpha=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 数据集
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from torchvision import transforms
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    train_size = int(0.9 * len(db_train))
    val_size = len(db_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(db_train, [train_size, val_size])

    # logging.info("=" * 60)
    # logging.info(f"训练集总数: {len(train_dataset)} 张")
    # logging.info(f"验证集总数: {len(val_dataset)} 张")
    # logging.info(f"批次大小 batch_size: {batch_size}")
    # logging.info(f"每轮训练迭代数 iterations/epoch: {len(train_dataset)//batch_size}")
    # logging.info("=" * 60)

    def worker_init_fn(worker_id):
        import random
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    train_loss_list = []
    val_loss_list = []

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.0001,
        save_path=os.path.join(snapshot_path, 'best_model.pth')
    )

    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        model.train()
        total_train_loss = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            optimizer.zero_grad()
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            total_train_loss += loss.item()

            iter_num += 1
            logging.info(
                f"[Iter {iter_num}] "
                f"Batch Loss = {loss.item():.6f} | "
                f"LR = {lr_:.6f}"
            )

        avg_train_loss = total_train_loss / len(trainloader)
        train_loss_list.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

                total_val_loss += loss.item() * image_batch.size(0)
                val_count += image_batch.size(0)

        avg_val_loss = total_val_loss / val_count
        val_loss_list.append(avg_val_loss)

        logging.info("-" * 60)
        logging.info(f"Epoch {epoch_num+1:3d} 结束 | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")
        logging.info("-" * 60)

        np.savetxt(os.path.join(snapshot_path, 'train_loss.txt'), train_loss_list, fmt='%.6f')
        np.savetxt(os.path.join(snapshot_path, 'val_loss.txt'), val_loss_list, fmt='%.6f')

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info("===== 早停触发，训练结束 =====")
            break

    plot_loss_curve(
        train_loss_list,
        val_loss_list,
        save_path=os.path.join(snapshot_path, 'loss_curve.png')
    )

    logging.info("\n" + "=" * 60)
    logging.info("训练完成！所有文件保存在路径：")
    logging.info(f"--> {snapshot_path}")
    logging.info("=" * 60)

    writer.close()
    return "Training Finished!"