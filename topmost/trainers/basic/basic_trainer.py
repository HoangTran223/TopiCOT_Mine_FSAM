import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from topmost.utils import static_utils
from topmost.models.basic.CombinedTM import CombinedTM
import wandb
import logging
import os
import scipy
import torch.optim

# Thêm
from topmost.trainers.SAM_function.SAM import SAM
# from topmost.trainers.SAM_function.FSAM import FSAM
from topmost.trainers.SAM_function.LookaheadSAM import AOSAM

# Thêm
# from pytorch_lightning import LightningModule

# Thêm
# from torch.cuda.amp import autocast, GradScaler
# from topmost.trainers.SAM_function.bypass_bn import enable_running_stats, disable_running_stats

class BasicTrainer():
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, rho=0.05, device = 'cuda', delta=0.3, mut = 0, sigmat = 1e-10, k1= 0.2, k2=0.4):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.rho = rho 
        self.device = device
        # Them
        self.logger = logging.getLogger('main')

        # Them
        self.mut = 0.0
        self.sigmat = 1e-10
        self.delta = delta
        self.k1 = k1 
        self.k2 = k2
        self.optimizer = self.make_aosam_optimizer()


    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                for group in self.optimizer.param_groups
                for p in group["params"] if p.grad is not None]),
            p=2)
        return norm

    
    # def make_sam_optimizer(self,):
    #     base_optimizer = torch.optim.SGD
    #     # FSAM
    #     optimizer = FSAM(
    #         self.model.parameters(),
    #         base_optimizer,
    #         device=self.device,
    #         lr=self.learning_rate,
    #         rho=self.rho,
    #         sigma=self.sigma,
    #         lmbda=self.lmbda) 

    #     return optimizer


    def make_aosam_optimizer(self,):
        base_optimizer = torch.optim.SGD
        # FSAM
        optimizer = AOSAM(
            self.model.parameters(),
            base_optimizer,
            device=self.device,
            lr=self.learning_rate,
            rho=self.rho,
            delta=self.delta,
            k1=self.k1,
            k2=self.k2) 

        return optimizer


    def make_adam_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer


    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    
    def train(self, dataset_handler, verbose=False):

        adam_optimizer = self.make_adam_optimizer()
        aosam_optimizer = self.make_aosam_optimizer()  

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(adam_optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)
        total_batches = len(dataset_handler.train_dataloader)
        T = self.epochs * total_batches

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            wandb.log({'epoch': epoch})

            for batch_idx, batch_data in enumerate(dataset_handler.train_dataloader):

                t = (epoch - 1) * total_batches + batch_idx + 1

                rst_dict = self.model(batch_data, epoch_id=epoch, batch_idx=batch_idx)
                batch_loss = rst_dict['loss']
                batch_loss.backward()
                
                grad_norm = self._grad_norm()

                # Tính mut, sigmat
                self.mut = self.delta * self.mut + (1 - self.delta) * grad_norm.item()**2
                self.sigmat = self.delta * self.sigmat + (1 - self.delta) * ((grad_norm.item()**2) - self.mut)**2
                
                # Tính c_t
                c_t = (t / T) * self.k1 + (1 - (t / T)) * self.k2

                if grad_norm.item()**2 >= (self.mut + c_t * (self.sigmat**0.5)):

                    aosam_optimizer.first_step(zero_grad=True)

                    rst_dict_adv = self.model(batch_data, epoch_id=epoch, batch_idx=batch_idx)
                    batch_loss_adv = rst_dict_adv['loss']
                    batch_loss_adv.backward()

                    aosam_optimizer.second_step(zero_grad=True)
                
                else:
                    adam_optimizer.step()
                    adam_optimizer.zero_grad()



    # def train(self, dataset_handler, device, verbose=False):
    #     optimizer = self.make_optimizer()
    #     # base_optimizer = torch.optim.SGD
    #     # optimizer = SAM(self.model.parameters(), base_optimizer, rho=0.05, adaptive=False)

    #     if self.lr_scheduler:
    #         print("===>using lr_scheduler")
    #         self.logger.info("===>using lr_scheduler")
    #         lr_scheduler = self.make_lr_scheduler(optimizer)

    #     data_size = len(dataset_handler.train_dataloader.dataset)

    #     for epoch in tqdm(range(1, self.epochs + 1)):
    #         self.model.train()
    #         loss_rst_dict = defaultdict(float)
    #         wandb.log({'epoch': epoch})

    #         for batch_idx, batch_data in enumerate(dataset_handler.train_dataloader):
                
    #             batch_data = {key: value.to(device) for key, value in batch_data.items()}

    #             # enable_running_stats(self.model)
    #             rst_dict = self.model(batch_data, epoch_id=epoch, batch_idx=batch_idx)
    #             batch_loss = rst_dict['loss']
    #             optimizer.zero_grad()

    #             # batch_loss.mean().backward()
    #             batch_loss.backward()
    #             optimizer.first_step(zero_grad=True)

    #             # disable_running_stats(self.model)
    #             rst_dict_adv = self.model(batch_data, epoch_id=epoch, batch_idx=batch_idx)
    #             batch_loss_adv = rst_dict_adv['loss']
    #             batch_loss_adv.mean().backward()
    #             optimizer.second_step(zero_grad=True)

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            for key in loss_rst_dict:
                wandb.log({key: loss_rst_dict[key] / data_size})

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)
                self.logger.info(output_log)

    def test(self, input_data):
        if not isinstance(self.model, CombinedTM):
            data_size = input_data.shape[0]
            theta = list()
            all_idx = torch.split(torch.arange(data_size), self.batch_size)

            with torch.no_grad():
                self.model.eval()
                for idx in all_idx:
                    batch_input = input_data[idx]
                    batch_theta = self.model.get_theta(batch_input)
                    theta.extend(batch_theta.cpu().tolist())
        else:
            data_size = input_data[0].shape[0]
            theta = list()
            all_idx = torch.split(torch.arange(data_size), self.batch_size)

            with torch.no_grad():
                self.model.eval()
                for idx in all_idx:
                    batch_bow = input_data[0][idx]
                    batch_contextual = input_data[1][idx]
                    batch_theta = self.model.get_theta(batch_bow, batch_contextual)
                    theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        if not isinstance(self.model, CombinedTM):
            train_theta = self.test(dataset_handler.train_data)
            test_theta = self.test(dataset_handler.test_data)
        else:
            train_theta = self.test((dataset_handler.train_data, dataset_handler.train_contextual_embed))
            test_theta = self.test((dataset_handler.test_data, dataset_handler.test_contextual_embed))
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings



