import gc
import thop
import time
import torch
import logging
import warnings
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
from datetime import timedelta

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
from torchmetrics import Accuracy, AveragePrecision
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from prettytable import PrettyTable

from melbanks import LogMelFilterBanks         # python hw_1/main.py
from yes_no_dataset import YesNoDataset, collate_fn
from cnn import CNN
    

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class YesNoDataModule(LightningDataModule):
    def __init__(self, ds_path: Union[str, Path], num_workers: int = 0, batch_size: int = 8):
        """YesNoDataModule.

        Args:
            ds_path: path to the dataset
            num_workers: number of CPU workers
            batch_size: number of sample in a batch

        """
        super().__init__()

        self._ds_path = ds_path
        self._num_workers = num_workers
        self._batch_size = batch_size

        # Raw waveforms are returned; feature extraction happens inside the model.
        self.train_transform = nn.Identity()
        self.valid_transform = nn.Identity()
        self.test_transform = nn.Identity()

    def prepare_data(self):
        """Download dataset if not present."""
        save_path = Path(self._ds_path)
        if not save_path.is_dir():
            SPEECHCOMMANDS(
                root=save_path,
                download=True,
            )

    def train_dataloader(self):
        log.info("Training data loaded.")
        dataset = YesNoDataset(root=self._ds_path, subset='training', transforms=self.train_transform)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        log.info("Validation data loaded.")
        dataset = YesNoDataset(root=self._ds_path, subset='validation', transforms=self.valid_transform)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        log.info("Testing data loaded.")
        dataset = YesNoDataset(root=self._ds_path, subset='testing', transforms=self.test_transform)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def print_class_distribution(self):
        from collections import Counter
        id2label = {v: k for k, v in YesNoDataset._LABELS.items()}
        for subset in ('training', 'validation', 'testing'):
            dataset = YesNoDataset(root=self._ds_path, subset=subset)
            counts = Counter(item['label'] for item in dataset)
            total = sum(counts.values())
            dist = {id2label[k]: v for k, v in sorted(counts.items())}
            print(f"{subset:>12}: {dist}  (total={total})")


class ClassificationModel(LightningModule):
    def __init__(self, n_mels=80, num_classes: int = 2, lr: float = 1e-3, conv_groups=1):
        super().__init__()

        self.lr = lr
        self.feature_extractor = LogMelFilterBanks(n_mels=n_mels)
        self.augment = nn.Identity()
        
        out_channels = 64
        self.encoder = CNN(in_channels=n_mels, out_channels=out_channels, conv_groups=conv_groups)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, num_classes, bias=True)
        )

        self.loss_func = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task='binary')
        self.valid_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.valid_prauc = AveragePrecision(task='binary')
        self.test_prauc = AveragePrecision(task='binary')

    def measure_flops(self):
        import copy
        n_mels = self.feature_extractor.n_mels
        encoder_cpu = copy.deepcopy(self.encoder).cpu()
        input_tensor = torch.randn(1, n_mels, 100, dtype=torch.float32)
        flops, params = thop.profile(encoder_cpu, inputs=(input_tensor,), verbose=False)
        del encoder_cpu
        return flops, params
        
    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.feature_extractor(x)

        if self.training:
            x = self.augment(x)

        x = self.encoder(x)

        return self.classifier(x)

    def loss(self, logits, labels):
        return self.loss_func(input=logits.squeeze(1), target=labels.float())

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch['waveforms'], batch['labels']
        bs = waveforms.size(0)
        
        y_logits = self.forward(waveforms)

        train_loss = self.loss(y_logits, labels)
        self.log("train_loss", train_loss, prog_bar=True, batch_size=bs)

        probs = torch.sigmoid(y_logits.squeeze(1))
        self.log("train_acc", self.train_acc(probs, labels), prog_bar=True, batch_size=bs)

        return train_loss

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch['waveforms'], batch['labels']
        bs = waveforms.size(0)
        y_logits = self.forward(waveforms)

        val_loss = self.loss(y_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, batch_size=bs)

        probs = torch.sigmoid(y_logits.squeeze(1))
        self.log("val_acc", self.valid_acc(probs, labels), prog_bar=True, batch_size=bs)
        self.valid_prauc.update(probs, labels)

    def on_validation_epoch_end(self):
        self.log("val_prauc", self.valid_prauc.compute(), prog_bar=True)
        self.valid_prauc.reset()

    def test_step(self, batch, _batch_idx):
        waveforms, labels = batch['waveforms'], batch['labels']
        bs = waveforms.size(0)
        y_logits = self.forward(waveforms)

        probs = torch.sigmoid(y_logits.squeeze(1))
        self.log("test_acc", self.test_acc(probs, labels), prog_bar=True, batch_size=bs)
        self.test_prauc.update(probs, labels)

    def on_test_epoch_end(self):
        self.log("test_prauc", self.test_prauc.compute())
        self.test_prauc.reset()
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        # scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], []


def train(config: dict):
    
    datamodule = YesNoDataModule(
        ds_path=config['ds_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    # datamodule.print_class_distribution()
    
    logger = TensorBoardLogger(
        save_dir=config['save_dir'],
        name=config['name'],
        version=config['version']
    )
    
    model = ClassificationModel(
        num_classes=1, 
        n_mels=config['n_mels'], 
        lr=config['lr'], 
        conv_groups=config['conv_groups']
    )

    flops, params = model.measure_flops()
    
    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-model",
        verbose=False,
    )
    
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True),
        checkpoint,
    ]
    
    trainer = Trainer(
        accelerator="auto",
        max_epochs=config['max_epochs'],
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        # enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    print(model.encoder)
    print(model.classifier)
    exit(2)
    start = time.time()
    trainer.fit(model, datamodule=datamodule)
    end = time.time()
    
    eval_res = trainer.test(ckpt_path=checkpoint.best_model_path, datamodule=datamodule, verbose=False)
     
    avg_epoch_time = (end - start) / (trainer.current_epoch + 1)
    results = {
        **eval_res[0],
        'flops(M)': round(flops / 10**6, 3),
        'params(K)': round(params / 10**3, 2),
        'train_time': timedelta(seconds=avg_epoch_time),
    }
    return results


def task3():
    # ============== task 3
    seed_everything(42)
    
    config = {
        'n_mels': None,         # defined later
        'lr': 1e-3,
        'ds_root': 'data', 
        'max_epochs': 50,
        'batch_size': 32,
        'num_workers': 4,
        'save_dir': 'runs',
        'name': '',
        'version': None,        # defined later
        'conv_groups': 1,
    }
    
    metrics = []
 
    # run experiments
    _NMELS = [20, 40, 60, 80]
    for i, n_mels in enumerate(_NMELS):
        print(f' EXP {i+1}/{len(_NMELS)} '.center(100, '='))
        newConfig = config.copy()
        newConfig['n_mels'] = n_mels
        newConfig['version'] = f'n_mels={n_mels}'
        results = train(newConfig)
        results['n_mels'] = n_mels
        metrics.append(results)
        
        gc.collect()
    
    table = PrettyTable()
    table.field_names = list(metrics[0].keys())
    
    # plot results table
    for x in metrics:
        table.add_row(list(x.values()))
    print(table)

    # generate plots
    fig, ax = plt.subplots()
    mels = [x['n_mels'] for x in metrics]
    test_acc = [x['test_acc'] for x in metrics]
    ax.scatter(mels, test_acc)
    ax.set_xlabel("n_mels")
    ax.set_ylabel("test_acc")
    ax.set_title("N_Mels vs Test Accuracy")
    ax.grid()
    fig.savefig('assets/hw_1/nmels_vs_acc.png')
    
    print('Done')
    

def task4():
    # ============== task 4
    seed_everything(42)
    
    config = {
        'n_mels': 64,
        'lr': 1e-3,
        'ds_root': 'data', 
        'max_epochs': 50,
        'batch_size': 32,
        'num_workers': 4,
        'save_dir': 'runs',
        'name': '',
        'version': None,        # defined later
        'conv_groups': None,    # defined later
    }
    metrics = []
    
    _GROUPS = [1, 2, 4, 8, 16]
    for i, group in enumerate(_GROUPS):
        print(f' EXP {i+1}/{len(_GROUPS)} '.center(100, '='))
        newConfig = config.copy()
        newConfig['conv_groups'] = group
        newConfig['version'] = f'groups={group}'
        
        results = train(newConfig)
        results['groups'] = group
        metrics.append(results)
    
    table = PrettyTable()
    table.field_names = list(metrics[0].keys())
    
    # plot results table
    for x in metrics:
        table.add_row(list(x.values()))
    print(table)
    
    groups = [x['groups'] for x in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    test_acc = [x['test_acc'] for x in metrics]
    axes[0].scatter(groups, test_acc)
    axes[0].set_xlabel("groups")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Conv Groups vs Test Accuracy")
    axes[0].grid()

    train_time = [x['train_time'].total_seconds() for x in metrics]
    axes[1].scatter(groups, train_time)
    axes[1].set_xlabel("groups")
    axes[1].set_ylabel("time(sec)")
    axes[1].set_title("Conv Groups vs Training Epoch Time")
    axes[1].grid()

    flops = [x['flops(M)'] for x in metrics]
    axes[2].scatter(groups, flops)
    axes[2].set_xlabel("groups")
    axes[2].set_ylabel("FLOPs(M)")
    axes[2].set_title("Conv Groups vs FLOPs")
    axes[2].grid()

    params = [x['params(K)'] for x in metrics]
    axes[3].scatter(groups, params)
    axes[3].set_xlabel("groups")
    axes[3].set_ylabel("Params(K)")
    axes[3].set_title("Conv Groups vs Params")
    axes[3].grid()

    fig.tight_layout()
    fig.savefig('assets/hw_1/conv_groups.png')
    print('Done')


if __name__ == '__main__':
    # task3()
    task4()
