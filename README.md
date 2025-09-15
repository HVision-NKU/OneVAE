# OneVAE: Unified Repository for Continuous and Discrete VAE Training
_Also the official open-source implementation of our work **OneVAE**._

📄 **Paper**: [OneVAE: Joint Discrete and Continuous Optimization Helps Discrete VAE Train Better](https://arxiv.org/abs/2508.09857)

**Key Contributions:**
1. **Multiple Structural Improvements** — Introduces several architecture-level enhancements for discrete VAE to boost reconstruction quality under high compression.  
2. **Progressive Training with Pretrained Continuous VAE** — Initializes from a high-quality pretrained continuous VAE and gradually transitions to discrete VAE, effectively leveraging strong priors.  
3. **Unified Model** — Achieves superior performance on both continuous and discrete representations within a single model.  

<img width="1665" height="607" alt="image" src="https://github.com/user-attachments/assets/b20c4cd8-985c-4127-b2de-08f33b2c5954" />

## Development Status
In addition to releasing the code of this work, we aim to provide a unified repository that supports fine-tuning and training of [multiple pretrained VAE models](#planned-supported-fine-tuning), enabling the community to better adapt VAEs to their specific needs.  
We are actively organizing and refining the codebase, and ⚡ **most features and resources will be released within two weeks!**

### Open Source Model
| Model Name   | Encoding Method                | Compression Ratio | Download Link                      |
| ------------ | ------------------------------ | ----------------- | ---------------------------------- |
| **OneVAE**   |  Discrete, Multi-Token Quant = 2     | 8 x 16 x 16    | [Link](https://huggingface.co/YupengZhou/OneVAE/tree/main) |
| **OneVAE**   |  Discrete, Multi-Token Quant = 2     | 16 x 16 x 16    | [Link](https://huggingface.co/YupengZhou/OneVAE/tree/main) |
| **OneVAE**   |  Discrete, Multi-Token Quant = 2     | 8 x 8 x 8    | Link |

## Visual Results

### Video Gallery
| Video1 | Video2  | 
| --- | --- |
|<video src="https://github.com/user-attachments/assets/e02eec54-5d83-420a-bcf2-caf10d9a0ef6" width=480>  |  <video src="https://github.com/user-attachments/assets/51e1abfa-139e-4ec0-af5c-c2422d254e3d" width=480> | 


### More Discrete Video Results on High-Compression VAE (4×16×16)
| Video1 | Video2  | Video3  | 
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/8db54421-ad3f-4526-bab7-a1737bfbaf14" width=480> |  <video src="https://github.com/user-attachments/assets/d2cbf790-f751-4cf2-a79d-8ecbdf9802a4" width=480> |  <video src="https://github.com/user-attachments/assets/f8aed5ab-97dd-4f42-9fe0-1ad18d77ca96" width=480> |

---

## Planned Supported Fine-Tuning

### Image VAE
- **FluxVAE**
- **LlamaGen**
- **SD-VAE**

### Video VAE
- **OneVAE (ours)**
- **WanVAE** _(Alibaba)_
- **HunyuanVideo VAE** _(Tencent)_

---

## TODO 
- [ ] Release model code (to be completed within two weeks)
- [ ] Provide pretrained weights download links
- [ ] Support additional types of VAE models

## LICENSE

The code is licensed under the Apache License 2.0. When using our repository to fine-tune other models, you must comply with the licenses of the respective pretrained models.
