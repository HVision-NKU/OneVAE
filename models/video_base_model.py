import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from util import instantiate_from_config
from models.model_utils.utils import flatten_state_dict
from modules.vqvae.enc_dec_llamagen import  CausalConv3d
# import the following quantizers 
from modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from modules.vqvae.quantize_simvq import VectorQuantizer2 as SimVectorQuantizer
from modules.vqvae.quantize_IBQ import IndexPropagationQuantize as IBQ
from modules.vqvae.quantize_FSQ import FSQ
# import the ema model
from models.model_utils.ema import LitEma
import os



class VideoBaseModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
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
        # load the vae model
        self.model = instantiate_from_config(vaeconfig)
        # if lossconfig is not None:
        #     self.loss = instantiate_from_config(lossconfig)
        # set the show_usage flag
        self.show_usage = show_usage
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))


        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        # load the quantizer
        self.quant_type = quant_type
        assert self.quant_type in ["SimVQ", "IBQ", "FSQ", "VectorQuantizer"], "quant_type must be one of ['SimVQ', 'IBQ', 'FSQ', 'VectorQuantizer']"
        if self.quant_type == "SimVQ":
            self.quantize = SimVectorQuantizer(**quant_config)
        elif self.quant_type == "IBQ":
            self.quantize = IBQ(**quant_config)
        elif self.quant_type == "FSQ":
            self.quantize = FSQ(**quant_config)
        else:
            self.quantize = VectorQuantizer(**quant_config)
        if not linear_quant_conv:
            raise NotImplementedError("linear_quant_conv must be True")
        else:
            self.quant_conv = torch.nn.Linear(ddconfig["z_channels"], embed_dim)
            self.post_quant_conv = torch.nn.Linear(embed_dim, ddconfig["z_channels"])
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if ema_decay is not None:
            self.use_ema = True
            if not hasattr(self, 'ema_encoder'):
                self.ema_model = LitEma(self.model, ema_decay)
                self.ema_quantize = LitEma(self.quantize, ema_decay)
                self.ema_quant_conv = LitEma(self.quant_conv, ema_decay)
                self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay)
        else:
            self.use_ema = False

    def process_state_dict(self,path,init_using_ema = True,ignore_keys=[]):
        print("Loading from checkpoint")
        sd = torch.load(path, map_location="cpu")
        if "discriminator" in ignore_keys:
            ignore_keys.remove("discriminator")
            for key in sd.keys():
                if "discriminator" in key:
                    ignore_keys.append(key)
        # 获取模型权重
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
                f.write("\nreplace_ema_sd:\n")
                f.write("\n".join(ema_sd.keys()) + "\n" if ema_sd else "None\n")
            print(f"Load weight information saved to {save_path}")
        if self.dirpath is not None:
            save_load_report()
        print(f"Restored from {path}")

    def encode(self, x, discrete: bool = True):   #  输入:  b,t,c,h,w
        """
        return
            when discrete==True:
                quant, emb_loss, info   # 可能是 Tensor，也可能是 list[Tensor]
            when discrete==False:
                h, None, None
        """

        assert hasattr(self.model, "input_shape")
        from einops import rearrange
        bsz = x.shape[0]
        x = rearrange(x, 'b t c h w -> ' + self.model.input_shape)
        h = self.model.encode(x)
        if not discrete:
            return h, None, None
        def _quantize_one(feat):
            """
            feat : Tensor, shape (b, c, t, h, w) 或展平后 (b*t, c, h, w)
            return: quant, emb_loss, info
            """
            if self.quant_conv is not None:
                if isinstance(self.quant_conv, CausalConv3d):
                    feat = self.quant_conv(feat)
                else:
                    feat = rearrange(feat, "b c t h w -> b t h w c")
                    feat = self.quant_conv(feat)
                    feat = rearrange(feat, "b t h w c -> b c t h w")

            # 4.2 如果还是 5 维，把 t 维展平成 batch 维
            if feat.dim() == 5:                       # (b, c, t, h, w)
                feat = feat.permute(0, 2, 1, 3, 4)    # -> (b, t, c, h, w)
                feat = feat.reshape(-1, *feat.shape[2:])  # (b*t, c, h, w)
            
            # 4.3 真正量化
            hw_shape = feat.shape[2:]
            quant, emb_loss, info = self.quantize(feat)
            temporal_shape = quant.shape[0] // bsz
            bt = (bsz,temporal_shape)
            return quant, emb_loss, info,bt,hw_shape

        if  not isinstance(h, list):
            h = [h]
        quant_list, emb_loss_list, info_list,bt_list,hw_shape_list = [], [], [],[],[]
        for h_i in h:
            q_i, e_i, info_i,bt_i,hw_shape_i = _quantize_one(h_i)
            quant_list.append(q_i)
            emb_loss_list.append(e_i)
            info_list.append(info_i)
            bt_list.append(bt_i)
            hw_shape_list.append(hw_shape_i)

        return quant_list, emb_loss_list, info_list,bt_list,hw_shape_list

    def decode(self, quant, bt=None,discrete=True):
        from einops import rearrange
        def _preprocess_one(q, bt):
            """
            q       : Tensor, shape (b*t, c, h, w) 或 (b, c, t, h, w) 或 (b, t, c, h, w)
            bt : None 或 (batch, t) 元组
            返回处理后的 Tensor，layout = self.model.input_shape
            """
            # --- 1. 复原 (b*t) -> (b, c, t, h, w) --------------------------------
            if bt is not None:
                bsz, tlen = bt
                q = q.reshape(bsz, tlen, *q.shape[1:])     # (b, t, c, h, w)
                q = rearrange(q, 'b t c h w -> b c t h w') # (b, c, t, h, w)

            # --- 2. post_quant_conv ------------------------------------------------
            if self.post_quant_conv is not None:
                if isinstance(self.post_quant_conv, CausalConv3d):
                    q = self.post_quant_conv(q)
                else:
                    q = rearrange(q, 'b c t h w -> b t h w c')
                    q = self.post_quant_conv(q)
                    q = rearrange(q, 'b t h w c -> b c t h w')

            # --- 3. 如果复原过 (b,c,t,h,w)，就再调回 (b,t,c,h,w) -------------------
            if bt is not None:
                q = rearrange(q, 'b c t h w -> b t c h w')

            # --- 4. 最终 rearrange 到模型需要的输入格式 ----------------------------
            q = rearrange(q, 'b t c h w -> ' + self.model.input_shape)
            return q
        if discrete:
            if isinstance(quant, list):
                # 处理 bt 是否也是 list 的情况
                bt_iter = bt if isinstance(bt, list) else [bt] * len(quant)
                quant = [
                    _preprocess_one(q_i, bt_i) for q_i, bt_i in zip(quant, bt_iter)
                ]
            else:  # quant 不是 list
                quant = _preprocess_one(quant, bt)
        if len(quant) == 1:
            dec = self.model.decode(quant[0])
        else:
            dec = self.model.decode(quant)
        # 再将dec从self.model.input_shape排列回b t c h w
        dec = rearrange(dec, self.model.input_shape + ' -> b t c h w')
        return dec

    def decode_from_codes(self, codes, bt=None,hw_shape=None, discrete=True):
        from einops import rearrange
        def _preprocess_one(q, bt,hw_shape):
            """
            q       : Tensor, shape (b*t, c, h, w) 或 (b, c, t, h, w) 或 (b, t, c, h, w)
            bt : None 或 (batch, t) 元组
            返回处理后的 Tensor，layout = self.model.input_shape
            """
            q = self.quantize.indices_to_codes(q)
            # print(f"q.shape: {q.shape}")
            # exit()
            # q = rearrange(q, 'b (h w) c -> b c h w', h=hw_shape[0], w=hw_shape[1])
            # --- 1. 复原 (b*t) -> (b, c, t, h, w) --------------------------------
            if bt is not None:
                bsz, tlen = bt
                q = q.reshape(bsz, tlen, *q.shape[1:])     # (b, t, c, h, w)
                q = rearrange(q, 'b t c h w -> b c t h w') # (b, c, t, h, w)

            # --- 2. post_quant_conv ------------------------------------------------
            if self.post_quant_conv is not None:
                if isinstance(self.post_quant_conv, CausalConv3d):
                    q = self.post_quant_conv(q)
                else:
                    q = rearrange(q, 'b c t h w -> b t h w c')
                    q = self.post_quant_conv(q)
                    q = rearrange(q, 'b t h w c -> b c t h w')

            # --- 3. 如果复原过 (b,c,t,h,w)，就再调回 (b,t,c,h,w) -------------------
            if bt is not None:
                q = rearrange(q, 'b c t h w -> b t c h w')

            # --- 4. 最终 rearrange 到模型需要的输入格式 ----------------------------
            q = rearrange(q, 'b t c h w -> ' + self.model.input_shape)
            return q
        if discrete:
            if isinstance(codes, list):
                # 处理 bt 是否也是 list 的情况
                bt_iter = bt if isinstance(bt, list) else [bt] * len(codes)
                hw_shape_iter = hw_shape if isinstance(hw_shape, list) else [hw_shape] * len(codes)
                quant = [
                    _preprocess_one(q_i, bt_i,hw_shape_i) for q_i, bt_i,hw_shape_i in zip(codes, bt_iter,hw_shape_iter)
                ]
            else:  # quant 不是 list
                quant = _preprocess_one(codes, bt,hw_shape)
        if len(quant) == 1:
            dec = self.model.decode(quant[0])
        else:
            dec = self.model.decode(quant)
        # 再将dec从self.model.input_shape排列回b t c h w
        dec = rearrange(dec, self.model.input_shape + ' -> b t c h w')
        return dec

    def forward(self, input, discrete=True):
        bsz = input.shape[0]
        discrete = self.discrete
        quant, diff, info, bt,hw_shape = self.encode(input,discrete=discrete)
        total_codes = [code.flatten().shape[0] for code in info]
        print(f"total_codes: {total_codes}")
        dec = self.decode_from_codes(info, bt=bt,hw_shape=hw_shape,discrete=discrete)
        dec = dec.reshape(input.shape)
        if self.training:
            return dec, diff[-1], info[-1]
        else:
            return dec, diff[-1]
    

    
    def on_train_batch_end(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        pass

    def validation_epoch_end(self, outputs):
        pass


    def get_last_layer(self):
        return self.model.get_last_layer()




