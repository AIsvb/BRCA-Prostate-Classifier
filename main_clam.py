from src.utils.reporting import create_pdf_from_folder, aggregate
from src.models.feature_aggregation.clam import CLAM_MB
from src.datasets.generic import Generic_MIL_Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from src.datasets.generic import save_splits
from sklearn.metrics import auc as calc_auc
from topk.svm import SmoothTop1SVM
from torch.amp import GradScaler
from src.utils.dataset import *
from torch import autocast
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import math
import csv
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_FREQ = 25

FEATURE_END_IDX = -2    # The top-left coordinates were concatenated to the features, hence the index

def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr, min_lr, epochs):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    # ------------------------------------------------------------------------------------------
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def count_params(model, train_only=False):
    if train_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = -1
        else:
            acc = float(correct) / count
        
        return acc, correct, count
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        configs:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_dir, ckpt_name):

        score = -val_loss

        self.save_checkpoint(val_loss, model, ckpt_dir / f'{ckpt_name}_latest.pt', False)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_dir / f'{ckpt_name}_best.pt', True)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_dir / f'{ckpt_name}_best.pt', True)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name, best=True):

        torch.save(model.state_dict(), ckpt_name)

        if best:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

def train_loop_clam(configs, epoch, model, loader, optimizer, writer, loss_fn, scaler):
    
    # -- convert to train mode
    model.train()
    
    # -- initialize variables for performance tracking
    acc_logger = Accuracy_Logger(configs.n_classes)
    inst_logger = Accuracy_Logger(configs.n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # -- loop over de dataset
    print('\n', flush=True)
    for batch_idx, (data, label) in enumerate(loader): 

        if batch_idx % configs.gc == 0 and configs.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(loader) + epoch, configs.warmup_epochs, configs.lr, configs.min_lr, configs.epochs)
           
        # -- move the data to the device
        data, label = data.to(device), label.to(device)
        
        # -- extract features (autocast prevents issues as a result of mixed precision)
        with autocast('cuda'):
            logits, _, Y_hat, _, instance_dict = model(data[:, :FEATURE_END_IDX], label=label, instance_eval=True)

        # -- compute error and log results
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = configs.bag_weight * loss + (1-configs.bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % LOG_FREQ == 0:
            print(f'batch {batch_idx}, loss: {loss_value:.4f}, instance_loss: {instance_loss_value:.4f}, ' + 
                f'weighted_loss: {total_loss.item():.4f}, label: {label.item()}, bag_size: {data.size(0)}')

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss /= configs.gc
        scaler.scale(total_loss).backward()

        # step
        if (batch_idx + 1) % configs.gc == 0:  
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(f'class {i} clustering acc {acc}: correct {correct}/{count}')

    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_clustering_loss:  {train_inst_loss:.4f}, train_error: {train_error:.4f}')
    for i in range(configs.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer and acc is not None:
            writer.add_scalar(f'train/class_{i}_acc', acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
 
def validate_clam(configs, split_idx, epoch, model, loader, writer, loss_fn, early_stopping = None):
    
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=configs.n_classes)
    inst_logger = Accuracy_Logger(n_classes=configs.n_classes)

    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), configs.n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):

            data, label = data.to(device), label.to(device)

            with autocast('cuda'):
                logits, Y_prob, Y_hat, _, instance_dict = model(data[:, :FEATURE_END_IDX], label=label, instance_eval=True)
            
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if configs.n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(configs.n_classes)])
        for class_idx in range(configs.n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print(f'\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}')
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(f'class {i} clustering acc {acc}: correct {correct}/{count}')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(configs.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        
        if writer and acc is not None:
            writer.add_scalar(f'val/class_{i}_acc', acc, epoch)
     
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_dir = Path(configs.results_dir), ckpt_name=f's_{split_idx}_checkpoint')
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_error
    else:
        torch.save(model.state_dict(), Path(configs.results_dir) / f's_{split_idx}_checkpoint_latest.pt')
    
    return False, val_error

def summary_clam(configs, model, loader):
    
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=configs.n_classes)

    test_error = 0.
    all_probs = np.zeros((len(loader), configs.n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            
            data, label = data.to(device), label.to(device)
        
            with autocast('cuda'):
                _, Y_prob, Y_hat, _, _ = model(data[:, :FEATURE_END_IDX])

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if configs.n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(configs.n_classes)])
        for class_idx in range(configs.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger

def train(datasets, split_idx, configs):
    '''
        train for a single split
    '''

    print(f'\nTraining split {split_idx}')

    # -- initialize tensorboard writer
    if configs.log_data:
        from tensorboardX import SummaryWriter
        writer_dir = configs.results_dir / f'tb_data_split_{str(split_idx)}'
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    # -- saving the data splits as csv
    print('\nInitialize train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], configs.results_dir / f'splits_{split_idx}.csv')
    print('Done!')
    print(f'Training on {len(train_split)} samples')
    print(f'Validating on {len(val_split)} samples')
    print(f'Testing on {len(test_split)} samples')

    # -- initialize the loss functions
    print('\nInitialize loss function...', end=' ')
    loss_fn = torch.nn.CrossEntropyLoss()
    print('Done!')

    instance_loss_fn = SmoothTop1SVM(n_classes = configs.n_classes).cuda() if device.type == 'cuda' else SmoothTop1SVM(n_classes = configs.n_classes)     
    configs.instance_loss_fn = instance_loss_fn

    # # -- initialize the aggregation model
    print('\nInitialize aggregation model...', end=' ')

    model = CLAM_MB(**{
        'instance_loss_fn': configs.instance_loss_fn,
        'embed_dim': configs.embed_dim,
        'size_arg': configs.size_arg,
        'subtyping': False,
        'k_sample': configs.k_sample,
        'gate': configs.gate,
        'n_classes': configs.n_classes,
        'dropout': configs.dropout
    })

    for param in model.parameters():
        param.requires_grad = True
        
    print(f'Total number of parameters: {count_params(model)}')
    print(f'Total number of trainable parameters: {count_params(model, True)}', flush=True)

    model.to(device)
    print('Done!')
    
    # -- initialize the AdamW optimizer
    print('\nInit optimizer ...', end=' ')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr, weight_decay=configs.wd)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, weighted = configs.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = False)
    test_loader = get_split_loader(test_split, testing = False)
    print('Done!')

    # -- set up early stopping
    print('\nSetup EarlyStopping...', end=' ')
    if configs.early_stopping:
        early_stopping = EarlyStopping(patience=configs.patience, stop_epoch=configs.stop_epoch, verbose = True)
    else:
        early_stopping = None
    print('Done!')

    # -- train and validate
    scaler = GradScaler('cuda')
    val_best = None
    for epoch in range(configs.epochs):
        train_loop_clam(configs, epoch, model, train_loader, optimizer, writer, loss_fn, scaler)
        
        # -- model selection step
        stop, val_error = validate_clam(configs, split_idx, epoch, model, val_loader, writer, loss_fn, early_stopping)

        if (val_best is None or val_error < val_best) and not configs.early_stopping:
            val_best = val_error
            torch.save(model.state_dict(), Path(configs.results_dir) / f's_{split_idx}_checkpoint_best.pt')

        if stop:
            break

    torch.save(model.state_dict(), configs.results_dir / f's_{split_idx}_checkpoint_latest.pt')

    results_dict, test_error, test_auc, acc_logger = summary_clam(configs, model, test_loader)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    _, val_error, val_auc, _= summary_clam(configs, model, val_loader)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    for i in range(configs.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error

def main(configs):

    # -- allow for selecting a subset from the precomputed splits
    if configs.k_start == -1:
        start = 0
    else:
        start = configs.k_start
    if configs.k_end == -1:
        end = configs.k
    else:
        end = configs.k_end

    # --
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    splits = np.arange(start, end)

    # set the learning rate (according to linear scaling law)
    if configs.lr is None or configs.lr < 0:  # only base_lr is specified
        configs.lr = configs.blr * configs.gc / 256
        
    print("base lr: %.2e" % (configs.lr * 256 / configs.gc))
    print("actual lr: %.2e" % configs.lr)

    for i in splits:
        datasets = dataset.return_splits(from_id=False, csv_path=f'{configs.split_dir}/split_{i}.csv')
        
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, configs)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        filename = os.path.join(configs.results_dir, 'split_{}_results.pkl'.format(i))
        with open(filename, 'wb') as file:
            pickle.dump(results, file)

    final_df = pd.DataFrame({'splits': splits, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(splits) != configs.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(configs.results_dir, save_name))   

    # -- automatic report construction
    aggregate(configs.results_dir).to_csv(configs.results_dir / 'eval_overview.csv')

    df = pd.read_csv(configs.results_dir / 'summary.csv')
    mean = pd.DataFrame([df[['test_auc', 'test_acc']].mean()], index=['Mean'])
    pd.concat([df, mean]).to_csv(configs.results_dir / 'summary.csv')

    create_pdf_from_folder(configs.results_dir)  

# -- parse command-line arguments
parser = argparse.ArgumentParser()

# - Experimental setup
parser.add_argument('--epochs',         type=int, default=20, help='Number of training epochs')
parser.add_argument('--warmup_epochs',  type=int, default=1, help='Number of warmup epochs')
parser.add_argument('--lr',             type=float, default=None, help='Learning rate')
parser.add_argument('--blr',            type=float, default=0.002, help='Base learning rate, will caculate the learning rate based on batch size')
parser.add_argument('--min_lr',         type=float, default=1e-6, help='Minimum learning rate')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels to use')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--lr_scheduler',   type=str, default='cosine', help='Learning rate scheduler', choices=['cosine', 'fixed'])
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiments')
parser.add_argument('--k', type=int, default=5, help='number of splits')
parser.add_argument('--k_start', type=int, default=-1, help='start split')
parser.add_argument('--k_end', type=int, default=-1, help='end split')
parser.add_argument('--weighted_sample', action='store_true', help='enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', help='implement early stopping')
parser.add_argument('--patience', type=int, default=20, help='patience of the Early Stopping algorithm')
parser.add_argument('--stop_epoch', type=int, default=50, help='minimum number of epochs to run with Early Stopping')
parser.add_argument('--gc', type=int, default=32, help='number of gradient accumulation steps')
parser.add_argument('--augmentation', type=str, default=None, help='which feature augmentations to use', choices=['macenko','20x + 40x + macenko', '20x + 40x', '20x', '40x', '20x + macenko', '40x + macenko'])

# - Input / Output
parser.add_argument('--split_dir', type=str, help='subfolder of the "splits" folder containing the splits for training')
parser.add_argument('--results_dir', type=str, default='results', help='results directory')
parser.add_argument('--log_data', action='store_true', help='log data using tensorboard')
parser.add_argument('--csv_path', type=str, help='path to the csv file with slide id / label pairs')
parser.add_argument('--data_dir', type=str, help='root directory containing either tile coordinates or extracted features')
parser.add_argument('--exp_code', type=str, help='tag that identifies the experiment (e.g. experiment_01)')

# - Aggregation model
parser.add_argument('--n_classes', type=int, default=2, help='number of classes of the classification problem')
parser.add_argument('--bag_weight', type=float, default=0.7, help='weight given to the bag loss (1-bag_weight for instance loss)')
parser.add_argument('--embed_dim', type=int, default=384, help='dimension of the input tile embeddings')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'], help='size indicator of clam model')
parser.add_argument('--gate', action='store_true', help='use gated attention network')
parser.add_argument('--k_sample', type=int, default=8, help='number of positive/negative patches to sample for instance level training')

# -- Run the cross-validated training
if __name__ == '__main__':
    
    configs = parser.parse_args()

    # -- set seeds
    seed_torch(configs.seed)

    # -- create dataset
    dataset = Generic_MIL_Dataset(csv_path = configs.csv_path,
                                  data_dir= configs.data_dir,
                                  shuffle = False,
                                  seed = configs.seed,
                                  print_info = True,
                                  label_dict = {'WILDTYPE':0, 'BRCA2':1},
                                  patient_strat=True,
                                  ignore=[])

    # -- formatting the path to the directory containing the splits
    configs.split_dir = Path(configs.split_dir)
    assert configs.split_dir.is_dir()

    # -- create experiment specific directory for saving the results
    configs.results_dir = Path(configs.results_dir) / f'{configs.exp_code}_s{configs.seed}'
    configs.results_dir.mkdir(parents=True, exist_ok=True)

    # -- store args as csv
    args_dict = vars(configs)
    with open(configs.results_dir / 'arguments.csv', mode='w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Argument', 'Value'])

        for key, value in args_dict.items():
            writer.writerow([key, value])

    results = main(configs)
    print('finished')
    print('end script')
