import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', use_hinge_loss=True, graph_loss_weight=0.3):
        self.n_epochs = n_epochs
        self.lr = lr
        # 设置设备：如果 CUDA 可用且 GPU 不为 None，使用 GPU；否则使用 CPU
        if torch.cuda.is_available() and GPU is not None:
            self.device = torch.device(f"cuda:{GPU}")
        else:
            self.device = torch.device("cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.use_hinge_loss = use_hinge_loss
        self.graph_loss_weight = graph_loss_weight

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        # 使用 AdamW，weight_decay=1e-4
        self.train_optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.to(self.device)
        
        # Scheduler: Warmup 5 epochs, then CosineAnnealing
        warmup_epochs = 5
        
        if self.n_epochs > warmup_epochs:
            # Linear Warmup
            scheduler1 = optim.lr_scheduler.LinearLR(self.train_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            # Cosine Annealing
            eta_min = self.lr * 0.35 
            scheduler2 = optim.lr_scheduler.CosineAnnealingLR(self.train_optimizer, T_max=self.n_epochs - warmup_epochs, eta_min=eta_min)
            self.scheduler = optim.lr_scheduler.SequentialLR(self.train_optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
        else:
            self.scheduler = optim.lr_scheduler.LinearLR(self.train_optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.n_epochs)

    def loss_fn(self, pred, label):
        # Ensure pred and label are 1D tensors
        if pred.dim() > 1:
            pred = pred.squeeze()
        if label.dim() > 1:
            label = label.squeeze()
        
        # Ensure pred and label are on the same device
        if pred.device != label.device:
            label = label.to(pred.device)
        
        # Create mask to filter NaN values
        mask = ~torch.isnan(label)
        
        # If all labels are NaN, return zero loss
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Apply mask and filter finite values
        pred_masked = pred[mask]
        label_masked = label[mask]
        
        # Ensure pred_masked and label_masked are finite
        pred_finite = torch.isfinite(pred_masked)
        label_finite = torch.isfinite(label_masked)
        valid_mask = pred_finite & label_finite
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred_masked = pred_masked[valid_mask]
        label_masked = label_masked[valid_mask]
        
        # Compute MSE Loss
        mse_loss = torch.mean((pred_masked - label_masked)**2)
        mse_loss = torch.clamp(mse_loss, 0.0, 1e6)  # Prevent numerical explosion
        
        return mse_loss

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        
        # Monitoring statistics collection (disabled)
        # gate_means = []
        # macro_atts = []
        # clamp_vals = []
        # attention_weights_list = []

        for data in data_loader:
            # DataLoader wraps batch as (1, N, T, F); squeeze to (N, T, F)
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm
            #########################

            model_output = self.model(feature.float())
            
            # 处理模型返回 tuple 的情况（如 hybermodel 返回 (pred, raw_scores, debug_stats)）
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                pred = model_output[0]  # 预测值
                raw_scores = model_output[1]  # 图结构分数 [H, N, N] 或 [N, N]
                
                # Debug stats collection (monitoring disabled)
                # if len(model_output) >= 3 and isinstance(model_output[2], dict):
                #     debug_stats = model_output[2]
                #     ...
                
                # 1. Main Task Loss
                main_loss = self.loss_fn(pred, label)
                
                # 2. Graph Structure Loss (Graph Regularization)
                # 获取 Label 并计算相关性矩阵作为 Target Graph
                # label: [N]
                # target_adj: [N, N]
                # 如果 yi, yj 同号，值为 1；异号，值为 -1
                y_reshaped = label.view(-1, 1)
                target_adj = torch.mm(y_reshaped, y_reshaped.t())  # [N, N] 收益率协方差矩阵的简化版
                target_adj = torch.sign(target_adj)  # 只关注方向一致性
                
                # 确保 target_adj 在正确的设备上
                target_adj = target_adj.to(raw_scores.device)
                
                # 处理 raw_scores 的形状
                # raw_scores 可能是 [H, N, M] (多头) 或 [N, M] (单头)
                # 其中 M = N+2 (包含2个宏观节点 + N个股票节点)
                # 我们需要切掉前2列（宏观列），只保留股票-股票部分 [N, N]
                if raw_scores.dim() == 3:
                    # 多注意力头：取平均，然后切掉前2列
                    avg_scores = torch.mean(raw_scores, dim=0)  # [N, M] where M = N+2
                    if avg_scores.shape[1] == avg_scores.shape[0] + 2:
                        # 如果列数 = 行数 + 2，说明包含2个宏观节点，切掉前2列
                        avg_scores = avg_scores[:, 2:]  # [N, N] - 只保留股票-股票部分
                elif raw_scores.dim() == 2:
                    # 单头：直接使用，但需要切掉前2列
                    if raw_scores.shape[1] == raw_scores.shape[0] + 2:
                        # 如果列数 = 行数 + 2，说明包含2个宏观节点，切掉前2列
                        avg_scores = raw_scores[:, 2:]  # [N, N] - 只保留股票-股票部分
                    elif raw_scores.shape[1] == raw_scores.shape[0]:
                        # 如果已经是 [N, N]，直接使用（兼容其他模型）
                        avg_scores = raw_scores
                    else:
                        # 形状不匹配，跳过 graph loss
                        avg_scores = None
                else:
                    # 如果维度不匹配，跳过 graph loss
                    avg_scores = None
                
                # Graph Loss: 根据use_hinge_loss选择使用Hinge Loss或MSE Loss
                if avg_scores is not None and avg_scores.shape == target_adj.shape:
                    # 确保 avg_scores 和 target_adj 都是有限值
                    scores_finite = torch.isfinite(avg_scores)
                    adj_finite = torch.isfinite(target_adj)
                    valid_graph_mask = scores_finite & adj_finite
                    
                    if valid_graph_mask.sum() > 0:
                        # 取出有效数据
                        t_valid = target_adj[valid_graph_mask]  # +1 或 -1
                        s_valid = avg_scores[valid_graph_mask]  # Logits
                        
                        if self.use_hinge_loss:
                            # === Hinge Loss：只惩罚方向错误，不惩罚过度自信 ===
                            # 逻辑：我们希望 target * score > margin
                            # 如果 target=1, score应该 > 0.1; 如果 target=-1, score应该 < -0.1
                            # 只要方向对了且有一点点信心，就不惩罚。允许 score 很大。
                            margin = 0.1
                            hinge = torch.relu(margin - t_valid * s_valid)
                            graph_loss = torch.mean(hinge)
                        else:
                            # === MSE Loss：消融实验，使用MSE替代Hinge Loss ===
                            # 将target_adj从{-1, +1}转换为{0, 1}用于MSE
                            # target=+1 -> 1, target=-1 -> 0
                            t_mse = (t_valid + 1) / 2.0  # 将{-1, +1}映射到{0, 1}
                            # 将scores归一化到[0, 1]范围（使用sigmoid）
                            s_mse = torch.sigmoid(s_valid)
                            # 计算MSE
                            graph_loss = torch.mean((s_mse - t_mse) ** 2)
                        
                        # 确保数值稳定
                        graph_loss = torch.clamp(graph_loss, 0.0, 1e6)
                    else:
                        graph_loss = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
                    
                    loss = main_loss + self.graph_loss_weight * graph_loss
                    
                    # 最终检查：确保总损失非负且有限
                    loss = torch.clamp(loss, 0.0, 1e6)
                else:
                    # 如果形状不匹配，只使用主损失
                    loss = main_loss
            else:
                # 其他模型：只计算主损失
                if isinstance(model_output, tuple):
                    pred = model_output[0]
                else:
                    pred = model_output
                loss = self.loss_fn(pred, label)
            
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        avg_loss = float(np.mean(losses))
        return avg_loss

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            # DataLoader wraps batch as (1, N, T, F); squeeze to (N, T, F)
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_na(label)
            label = zscore(label)
                        
            model_output = self.model(feature.float())
            # 处理模型返回 tuple 的情况
            if isinstance(model_output, tuple):
                pred = model_output[0]
            else:
                pred = model_output
            loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))
    
    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        # Data should be PreprocessedTimeSeriesDataset object
        if not (hasattr(data, '__getitem__') and hasattr(data, 'get_index')):
            raise ValueError("Data must be a PreprocessedTimeSeriesDataset object")
        
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
        
            # 早停机制已关闭
            # if train_loss <= self.train_stop_loss_thred:
            #     best_param = copy.deepcopy(self.model.state_dict())
            #     torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
            #     break
        

    def predict(self, dl_test):
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            # DataLoader wraps batch as (1, N, T, F); squeeze to (N, T, F)
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            
            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                model_output = self.model(feature.float())
                # 处理模型返回 tuple 的情况
                if isinstance(model_output, tuple):
                    pred = model_output[0]
                else:
                    pred = model_output
                pred = pred.detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        # Get index from PreprocessedTimeSeriesDataset
        test_index = dl_test.get_index()
        predictions = pd.Series(np.concatenate(preds), index=test_index)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return predictions, metrics
