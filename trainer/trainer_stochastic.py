import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference_stochastic, sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, np_softmax


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)

            text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var = self.model(data, is_train=True)
            # [bs, dim], [bs, bs, dim], [bs, bs, dim], [bs, 1, dim], [bs, dim]

            output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)
            loss = self.loss(output, self.model.clip.logit_scale)

            # @WJM: support text embedding regulrization
            video_embeds_pooled_avg = torch.mean(video_embeds_pooled,dim=1).squeeze()
            pointer = video_embeds_pooled_avg - text_embeds
            text_support = pointer / pointer.norm(dim=-1, keepdim=True) * torch.exp(text_log_var) + text_embeds
            output_support = sim_matrix_training(text_support, video_embeds_pooled, self.pooling_type)
            loss_support = self.loss(output_support, self.model.clip.logit_scale)

            loss_all = loss + loss_support * self.config.support_loss_weight
            loss_all.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            if self.config.noloss_record:
                pass
            else:
                gen_log(model_path=self.config.model_path, log_name='log_tot_loss',
                        msg=loss_all.item())
                gen_log(model_path=self.config.model_path, log_name='log_ori_loss',
                        msg=loss.item())
                gen_log(model_path=self.config.model_path, log_name='log_sup_loss',
                        msg=loss_support.item())


            if batch_idx % self.log_step == 0:
                msg = ('Train Epoch: {} dl: {}/{} Total Loss: {:.6f}, Original Loss: {:.6f}, Support Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss_all.detach().item(),
                    loss.detach().item(),
                    loss_support.detach().item(),
                    ))
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)


            if batch_idx in eval_steps:

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

                else:
                    val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                    self.model.train()

                    if val_res['R1-window'] > self.best_window:
                        self.best_window = val_res['R1-window']
                        self._save_checkpoint(epoch, save_best=True)

                    if val_res['R1'] > self.best:
                        self.best = val_res['R1']

                    msg = (" Current Best Window Average R@1 is {}".format(self.best_window), " Current Best R@1 is {}\n\n".format(self.best))
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)

                text_embed, vid_embed, vid_embed_pooled, text_embed_stochastic = self.model(data, return_all_frames=True, is_train=False)

                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]

            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda()

            # build stochastic text embeds #########################################
            self.model.stochastic.cpu()
            start_selection_time = time.time()

            # initialize text_embeds_stochastic_allpairs: to avoid data leakage, break vid-txt dependence by dataloader
            text_embeds_stochastic_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))

            # @WJM: the principle is to use the query video to process text
            # sequential process to save memory:
            for (idx_vid, single_vid), single_vid_embed_pooled in tqdm(zip(enumerate(vid_embeds),vid_embeds_pooled)):

                single_vid_vec = single_vid.unsqueeze(0)
                # repeat as the same size of all texts
                single_vid_repeat = single_vid_vec.tile((text_embeds.shape[0], 1, 1)) # [bs_t, #F, dim]

                all_text_embed_stochstic = []
                for trial in range(self.config.stochasic_trials):
                    all_text_embed_stochastic, _, _ = self.model.stochastic(text_embeds, single_vid_repeat) # [bs_t, dim]
                    all_text_embed_stochstic.append(all_text_embed_stochastic)
                all_text_embed_stochstic_arr = torch.stack(all_text_embed_stochstic, dim=0) # [#trials, bs_t, dim]

                # normalization before compute cos-sim
                all_text_embed_stochstic_arr = all_text_embed_stochstic_arr / all_text_embed_stochstic_arr.norm(dim=-1, keepdim=True)
                single_vid_embed_pooled = single_vid_embed_pooled / single_vid_embed_pooled.norm(dim=-1, keepdim=True)

                # compute cos-sim
                sim_select = torch.sum(torch.mul(all_text_embed_stochstic_arr, single_vid_embed_pooled), dim=-1) # [#trial, bs_t]

                # find max cos, take idx
                max_indices = torch.argmax(sim_select, dim=0) # [bs_t]

                # select based on the idx
                selected_plane = torch.ones((all_text_embed_stochstic_arr.shape[1], all_text_embed_stochstic_arr.shape[2]))
                for i in range(all_text_embed_stochstic_arr.shape[1]):
                    selected_plane[i, :] = all_text_embed_stochstic_arr[max_indices[i], i, :]
                text_embeds_stochastic_allpairs[idx_vid,:,:] = selected_plane

            end_selection_time = time.time()
            msg = (f'To compute all stochastic-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time}\n')
            gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)
            self.model.stochastic.cuda()
            # finish build stochastic text embeds #########################################

            # @WJM: rm unnecessary tensor to release memory
            del text_embeds, vid_embeds
            gc.collect()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(text_embeds_stochastic_allpairs,
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)

            # @WJM: rm unnecessary tensor to release memory
            del text_embeds_stochastic_allpairs, vid_embeds_pooled
            gc.collect()

            # @WJM: can use light implementation to avoid memory OOM:
            if self.config.save_memory_mode:
                start_sims = time.time()
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg='Use sim_matrix_inference_stochastic_light()')
                sims = sim_matrix_inference_stochastic_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.config.batch_size_split, self.config)
                end_sims = time.time()
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=f'batch size split = {self.config.batch_size_split}, sims compute time={end_sims-start_sims}')

            else:
                sims = sim_matrix_inference_stochastic(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            # add DSL
            if self.config.DSL:
                sims = sims * np_softmax(sims*100, axis=0)


            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            msg = (f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  )
            gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

            res['loss_val'] =  total_val_loss

            return res
