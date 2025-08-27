import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from util import instantiate_from_config

from models.model_utils.utils import flatten_state_dict, save_video

from modules.vqvae.enc_dec_llamagen import Encoder, Decoder, CausalConv3d
# from modules.vqvae.enc_dec_llamagenv2 import Encoder as Encoder2, Decoder as Decoder2
# from modules.vqvae.enc_dec_llamagenv3 import Encoder as Encoder3, Decoder as Decoder3
# from modules.vqvae.enc_dec_llamagenv4 import Encoder as Encoder4, Decoder as Decoder4
# from modules.vqvae.enc_dec_llamagenv5 import Encoder as Encoder5, Decoder as Decoder5
# from modules.vqvae.movqgan_enc_dec_pz import Encoder as Encoder_8x8, Decoder as Decoder_8x8
from modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from modules.vqvae.quantize_simvq import VectorQuantizer2 as SimVectorQuantizer
from modules.vqvae.quantize_IBQ import IndexPropagationQuantize as IBQ
from modules.vqvae.quantize_FSQ import FSQ
from einops import rearrange

from models.model_utils.ema import LitEma

import os
import cv2
import numpy as np
from torch.distributed import barrier


from eval.tokenizer.cal_lpips import calculate_lpips
from eval.tokenizer.cal_ssim import calculate_ssim
from eval.tokenizer.cal_psnr import calculate_psnr

class ImageBaseModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ema_decay=None,
                 show_usage = True,
                 freezed_keys=[],
                 trained_keys=[],
                 dirpath=None,
                 quant_type=None,
                 quant_config=None,
                 linear_quant_conv = True,
                 discrete = True,
                 vaeconfig=None,
                 **kwargs
                 ):
        super().__init__()
        self.dirpath = dirpath
        self.n_embed = n_embed
        self.ema_decay = ema_decay
        self.discrete = discrete
        self.show_usage = show_usage
        self.quant_type = quant_type
        self.model = instantiate_from_config(vaeconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)

        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
        self.process_modules_gradient(freezed_keys,trained_keys)

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if self.quant_type is not None and self.quant_type == "SimVQ":
            self.quantize = SimVectorQuantizer(n_embed, embed_dim, beta=0.25,
                                            remap=remap, sane_index_shape=sane_index_shape)
        elif self.quant_type is not None and self.quant_type == "IBQ":
            self.quantize = IBQ(n_embed, embed_dim, beta=0.25,
                                            remap=remap)
        elif self.quant_type is not None and self.quant_type == "FSQ":
            self.quantize = FSQ(**quant_config)
        else:
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                            remap=remap, sane_index_shape=sane_index_shape)
        if not hasattr(self.model, 'quant_conv') or not hasattr(self.model, 'post_quant_conv'):
            if 'time_downsample' in ddconfig and not linear_quant_conv:
                stride = (1, 1, 1)
                kernel_size = (3, 1, 1)
                self.quant_conv = CausalConv3d(ddconfig["z_channels"], embed_dim,  time_stride=stride[0],kernel_size=kernel_size,padding =0)
                self.post_quant_conv = CausalConv3d(embed_dim, ddconfig["z_channels"],  time_stride=stride[0],kernel_size=kernel_size,padding = 0)
            else:
                self.quant_conv = torch.nn.Linear(ddconfig["z_channels"], embed_dim)
                self.post_quant_conv = torch.nn.Linear(embed_dim, ddconfig["z_channels"])
        else:
            # 如果模型已有这些属性，则使用模型自身的
            self.quant_conv = self.model.quant_conv
            self.post_quant_conv = self.model.post_quant_conv
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        if ema_decay is not None:
            self.use_ema = True
            # 判断是否有属性，有就不创建
            if not hasattr(self, 'ema_encoder'):
                self.ema_model = LitEma(self.model, ema_decay)
                self.ema_quantize = LitEma(self.quantize, ema_decay)
                self.ema_quant_conv = LitEma(self.quant_conv, ema_decay)
                self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay)
        else:
            self.use_ema = False
    def process_modules_gradient(self,freezed_keys,trained_keys):
        if not freezed_keys and not trained_keys:
            return
        if freezed_keys and trained_keys:
            raise ValueError("freezed_keys 和 trained_keys 不能同时设置为非空列表，请检查配置！")
        for name, param in self.encoder.named_parameters():
            full_name = f"encoder.{name}"
            if freezed_keys:
                param.requires_grad = not any(freeze_key in full_name for freeze_key in freezed_keys)
            elif trained_keys:
                param.requires_grad = any(train_key in full_name for train_key in trained_keys)
        # 设置解码器参数的训练状态
        for name, param in self.decoder.named_parameters():
            full_name = f"decoder.{name}"
            if freezed_keys:
                param.requires_grad = not any(freeze_key in full_name for freeze_key in freezed_keys)
            elif trained_keys:
                param.requires_grad = any(train_key in full_name for train_key in trained_keys)

        # 动态计算哪些参数被冻结和训练
        frozen_params = []
        trained_params = []
        for name, param in list(self.vae.named_parameters()) + list(self.loss.named_parameters()) :
            if not param.requires_grad:
                frozen_params.append(name)
            else:
                trained_params.append(name)
        if self.dirpath is not None:
        # 保存当前状态到文件
            save_dir = os.path.join(self.dirpath, "config")
            os.makedirs(save_dir, exist_ok=True)
            @rank_zero_only
            def save_parameter_status():
                save_path = os.path.join(save_dir, "parameter_status.txt")
                with open(save_path, "w") as f:
                    f.write("==== Parameter Training Status ====\n")
                    f.write(f"Freezed keys (calculated): {frozen_params}\n" if frozen_params else "Freezed keys: None\n")
                    f.write(f"Trained keys (calculated): {trained_params}\n" if trained_params else "Trained keys: None\n")

                print(f"Parameter training status saved to {save_path}")
            save_parameter_status()
    def process_state_dict(self,path,init_using_ema = True,ignore_keys=[]):
        print("Loading from checkpoint")
        sd = torch.load(path, map_location="cpu")
        if "discriminator" in ignore_keys:
            print("ignore discriminator")
            ignore_keys.remove("discriminator")
            for key in sd.keys():
                if "discriminator" in key:
                    ignore_keys.append(key)
        if isinstance(sd, dict) and "model" in sd and len(sd) == 1:
            sd = sd["model"]
            sd = flatten_state_dict(sd)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        ### 使用ema 权重初始化训练模型
        if init_using_ema:
            # 如果存在 EMA 权重,将其复制到普通权重
            old_ema_sd = {}
            ema_sd = {}
            for k, v in sd.items():
                if k.startswith('model_ema.'):
                    old_ema_sd[k.replace('model_ema.', '')] = v
                elif k.startswith('ema_'):
                    old_ema_sd[k.replace('ema_', '')] = v
                elif 'ema' in k:
                    old_ema_sd[k.replace('.ema', '')] = v
            for k, v in sd.items():
                if k.replace(".", "").replace("encoder","encoder.").replace("decoder","decoder.") in old_ema_sd:
                    ema_sd[k] = old_ema_sd[k.replace(".", "").replace("encoder","encoder.").replace("decoder","decoder.")]
            # 更新 sd 字典
            if ema_sd == {}:  # 明确检查是否为空字典
                print("No EMA weights found in checkpoint")
                raise ValueError("No EMA weights found in checkpoint")
            else:
                print("Found EMA weights, copying to main weights...")
                sd.update(ema_sd)
        else:
            ema_sd = None
               # 判断key 中是否包含ema
        ## 加载保存的ema权重 到ema 模型中
        if any("ema" in key for key in sd.keys()) and self.ema_decay is not None:
            self.use_ema = True
            ema_decay = self.ema_decay
            self.ema_model = LitEma(self.model, ema_decay)
            self.ema_quantize = LitEma(self.quantize, ema_decay)
            self.ema_quant_conv = LitEma(self.quant_conv, ema_decay)
            self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay)
        if self.quant_type is not None and (self.quant_type =="SimVQ" or self.quant_type =="FSQ"):
            if 'quantize.embedding.weight' not in sd:
                print("init weight do not have code")
            else:
                del sd['quantize.embedding.weight']
        else:
            old_embed = sd.get('quantize.embedding.weight', None)
            input_embeddings = self.quantize.embedding.weight.data
            print(type(old_embed), old_embed.shape)
            print("input_embeddings", type(input_embeddings), input_embeddings.shape)

            if input_embeddings.shape[0] != old_embed.shape[0]:
                old_token_n = old_embed.shape[0]
                if input_embeddings.shape[1] != old_embed.shape[1]:
                    old_embed_n = old_embed.shape[1]
                    input_embeddings[:old_token_n, :old_embed_n] = old_embed[:old_token_n, :old_embed_n]
                else:
                    input_embeddings[:old_token_n] = old_embed[:old_token_n]
                sd['quantize.embedding.weight'] = input_embeddings
                print("replace quantize.embedding.weight!")


        keys = list(sd.keys())
        for ik in ignore_keys:
            if 'weight' in ik:
                sd[ik.replace('_conv', '_3dconv')] = sd[ik].unsqueeze(2).repeat(1,1,3,1,1)
            elif 'bias' in ik:
                sd[ik.replace('_conv', '_3dconv')] = sd[ik]

        for ik in ignore_keys:
            print("Deleting key {} from state_dict.".format(ik))
            del sd[ik]

        missing_shapes=[]
        # 删除不匹配的参数
        for name, param in list(sd.items()):
            # 如果模型中存在该参数，但形状不匹配，则删除
            if name in self.state_dict() and self.state_dict()[name].shape != param.shape:
                del sd[name]  # 删除该参数
                missing_shapes.append(name)
        return sd,missing_shapes,ema_sd
    def init_from_ckpt(self, path, init_using_ema = False, ignore_keys=[]):
        sd,missing_shapes,ema_sd    = self.process_state_dict(path,init_using_ema,ignore_keys)
        # 加载 state_dict
        result = self.load_state_dict(sd, strict=False)

        # 获取缺失和多余的键
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
        @rank_zero_only
        def save_load_report():
            # 保存路径
            save_dir = os.path.join(self.dirpath, "config")
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(save_dir, "load_weight.txt")

            # 写入信息到文件
            with open(save_path, "w") as f:
                f.write("==== Load Weight Report ====\n")
                f.write("Missing keys (not loaded):\n")
                f.write("\n".join(missing_keys) + "\n" if missing_keys else "None\n")
                f.write("\nUnexpected keys (ignored):\n")
                f.write("\n".join(unexpected_keys) + "\n" if unexpected_keys else "None\n")
                f.write("\nmissing_shapes keys (ignored):\n")
                f.write("\n".join(missing_shapes) + "\n" if missing_shapes else "None\n")
                f.write("\nSuccessfully loaded weights for the following keys:\n")
                loaded_keys = [key for key in sd.keys() if key not in unexpected_keys]
                f.write("\n".join(loaded_keys) + "\n" if loaded_keys else "None\n")
                ### 记录一下ema_sd
                f.write("\nreplace_ema_sd:\n")
                f.write("\n".join(ema_sd.keys()) + "\n" if ema_sd else "None\n")
            print(f"Load weight information saved to {save_path}")
        if self.dirpath is not None:
            save_load_report()
        print(f"Restored from {path}")

    def encode(self, x, discrete=True):  # input: b,c,h,w
        assert hasattr(self.model, "input_shape")
        # 直接根据 input_shape 排列输入
        x = rearrange(x, 'b c h w -> ' + self.model.input_shape)
        h = self.model.encode(x)

        if discrete:
            if self.quant_conv is not None:
                # 直接按 image 处理
                h = rearrange(h, "b c h w -> b h w c")
                h = self.quant_conv(h)
                h = rearrange(h, "b h w c -> b c h w")

            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info
        else:
            return h, None, None

    def decode(self, quant, discrete=True):
        if discrete:
            if self.post_quant_conv is not None:
                quant = rearrange(quant, 'b c h w -> b h w c')
                quant = self.post_quant_conv(quant)
                quant = rearrange(quant, 'b h w c -> b c h w')

            quant = rearrange(quant, 'b c h w -> ' + self.model.input_shape)

        dec = self.model.decode(quant)
        # 再排列回 b c h w
        dec = rearrange(dec, self.model.input_shape + ' -> b c h w')
        return dec

    def forward(self, input, discrete=True):
        discrete = self.discrete
        # encode
        quant, diff, info = self.encode(input, discrete=discrete)
        # decode
        dec = self.decode(quant, discrete=discrete)
        dec = dec.reshape(input.shape)
        if self.training:
            return dec, diff, info
        else:
            return dec, diff

    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model(self.model)
            self.ema_quantize(self.quantize)
            if self.quant_conv is not None:
                self.ema_quant_conv(self.quant_conv)
            if self.post_quant_conv is not None:
                self.ema_post_quant_conv(self.post_quant_conv)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # 处理新的数据格式 (data_type, [idx, x])
        if isinstance(batch[0], str):  # DualDataLoader格式
            data_type, data = batch
            idx, x = data  # 解包内部的 [idx, x]
            if data_type == "image":
                x = x.unsqueeze(1)  # 添加时间维度 
        else:  # 原有格式 (idx, x)
            idx, x = batch
            data_type = None

        discrete = self.discrete
        xrec, qloss, info = self(x)
        
        if self.show_usage and self.discrete:
            if "codebook_used" not in dict(self.named_buffers()):
                self.register_buffer("codebook_used",  torch.zeros(65536, device="cuda"))
            min_encoding_indices = info[2]
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / (self.quantize.codebook_size if self.quant_type == "FSQ" else self.n_embed)

        if optimizer_idx == 0:
            # autoencode

            
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")


            if discrete:
                self.log("train/codebook_usage", codebook_usage, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        # 拆分 batch
        idx, x = batch  # idx 为索引，x 为数据
        x_original =x.detach().clone()
        xrec, qloss = self(x)
        # Calculate losses
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)


        xrec = xrec.reshape(x.shape)
        x_concat = torch.cat([x,xrec],dim = -1)
        # 返回包含克隆的 x
        return {
            "x_concat": x_concat,
            "index": idx,
            "original_x": x_original # 克隆后的 x
        }

    def validation_epoch_end(self, outputs):
        # 汇总所有 GPU 的数据
        if not isinstance( outputs[0], list):
            outputs = [outputs]

        for cur_res_outputs in outputs:
            flattened_outputs = cur_res_outputs

            all_x_concat = torch.cat([output["x_concat"] for output in flattened_outputs], dim=0)  # 合并 batch 数据
            all_x_concat = self.all_gather(all_x_concat)  # 跨设备聚合
            x = all_x_concat.reshape(-1,*all_x_concat.shape[2:])
            # 获取 all_indices
            all_indices = torch.cat([output["index"] for output in flattened_outputs], dim=0)
            all_indices = self.all_gather(all_indices)  # 跨设备聚合
            indices = all_indices.flatten().tolist()

            # 获取 original_x 的数据
            all_original_x = torch.cat([output["original_x"] for output in flattened_outputs], dim=0)
            all_original_x = self.all_gather(all_original_x)  # 跨设备聚合
            original_x = all_original_x.reshape(-1, *all_original_x.shape[2:])
            width = x.shape[-2]
            frame_nums = x.shape[-4]

            total_psnr = 0
            total_lpips = 0
            total_ssim = 0
            # 只在主进程处理
            if self.trainer.is_global_zero:
                    # 将张量保存到文件
                save_dir = os.path.join(self.trainer.checkpoint_callback.dirpath, "videos",  f"step_{self.global_step}",f"res_{width}_frame_{frame_nums}")
                os.makedirs(save_dir, exist_ok=True)
                print(f"val video save in {save_dir}")
                
                # 获取width（假设 x 是 [batch_size, channels, height, width*2]）
                half_width = x.shape[-1] // 2  # 计算宽度的一半，因为它是拼接的

                # 切分 x 为 gt 和 generated
                gt = x[..., :half_width]  # 真实视频部分
                generated = x[..., half_width:]  # 生成视频部分
                print(gt.shape,generated.shape,"calculating pips......")
                # 计算每个样本的指标
                total_lpips += calculate_lpips(generated, original_x, self.device)["avg"]
                total_psnr += calculate_psnr(generated.cpu(), original_x.cpu())["avg"]
                total_ssim += calculate_ssim(generated.cpu(), original_x.cpu())["avg"]

                for i, idx in enumerate(indices):  # Iterate over the batch
                    output_video_filename = os.path.join(save_dir, f"sample_{idx}.mp4")
                    # Save each sample's video
                    save_video(x[i], output_video_filename)


                # 计算平均指标（可选）
                num_samples = len(indices)
                avg_lpips = total_lpips 
                avg_psnr = total_psnr
                avg_ssim = total_ssim 

                # # 记录并保存这些指标
                self.log(f"val_score_res_{width}_frame_{frame_nums}/avg_lpips", avg_lpips, prog_bar=False, logger=True, on_step=False, on_epoch=True,rank_zero_only=True)
                self.log(f"val_score_res_{width}_frame_{frame_nums}/avg_psnr", avg_psnr, prog_bar=False, logger=True, on_step=False, on_epoch=True,rank_zero_only=True)
                self.log(f"val_score_res_{width}_frame_{frame_nums}/avg_ssim", avg_ssim, prog_bar=False, logger=True, on_step=False, on_epoch=True,rank_zero_only=True)
                # 打印 val 结束时的指标
                print(f"val end - avg_lpips: {avg_lpips:.4f}, avg_psnr: {avg_psnr:.4f}, avg_ssim: {avg_ssim:.4f}")
        barrier()

    def on_after_backward(self):
        # 计算所有参数梯度的平均值
        total_grad = 0
        param_count = 0

        for param in self.parameters():
            if param.grad is not None:
                total_grad += param.grad.data.norm(2)  # L2 范数
                param_count += 1

        if param_count > 0:
            avg_grad = total_grad / param_count
            self.log("avg_grad_norm", avg_grad)  # 将平均梯度记录到日志中
        else:
            self.log("avg_grad_norm", 0.0)  # 如果没有参数，则记录为 0

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.model.parameters())+
                                list(self.quantize.parameters())+
                                list(self.quant_conv.parameters())+
                                list(self.post_quant_conv.parameters()),
                                lr=lr, 
                                betas=(0.5, 0.9),
                                eps=1e-06,
                                weight_decay=0.1
                                )
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(),
            lr=lr, 
            betas=(0.5, 0.9),
            eps=1e-06,
            weight_decay=0.1
            )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.model.get_last_layer()




