import os
import torch
from omegaconf import OmegaConf
from argparse import ArgumentParser
from util import instantiate_from_config_and_ckpt,instantiate_from_config
from moviepy.editor import ImageSequenceClip
import cv2
import numpy as np
from pathlib import Path

def save_video(tensor, filename, fps=16, bitrate='8000k'):
    # tensor shape should be (t, c, h, w), assuming 3 channels (RGB)
    t, c, h, w = tensor.shape
    
    # Ensure the values are in the range [-1, 1] and convert to [0, 255]
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) * 127.5
    tensor = tensor.permute(0, 2, 3, 1)  # Change to (t, h, w, c) for moviepy
    
    # Convert tensor to uint8 type
    tensor = tensor.to(torch.uint8)
    
    # Convert tensor to a list of frames
    frames = [frame.cpu().numpy() for frame in tensor]
    
    # Create a video clip from the frames
    video_clip = ImageSequenceClip(frames, fps=fps)
    
    # Write the video to file with a specified bitrate 
    video_clip.write_videofile(filename, codec='libx264', bitrate=bitrate)


def run_eval(config, enable_ema=False, video_path=None, enable_bf16=False, enable_tile=True, no_segment=True, encode_method='default'):
    exp_name = config.get('exp_name', 'exp25_unify')
    use_continue = False
    # 设置输出目录
    output_root_dir = f"./test_dir/{exp_name}" + ("_continue" if use_continue else "")
    if enable_ema:
        output_root_dir = f"./test_dir/{exp_name}_ema" + ("_continue" if use_continue else "")
    os.makedirs(output_root_dir, exist_ok=True)
    config['model']['params']['dirpath'] = output_root_dir
    if "test_ckpt_path" in config:
        ckpt_path = config['test_ckpt_path']
        print(f"Loading checkpoint from {ckpt_path}")
        model = instantiate_from_config_and_ckpt(config['model'], ckpt_path).to("cuda")
    else:
        model = instantiate_from_config(config['model']).to("cuda")
    model.eval()
    if video_path:
        # 处理指定路径的视频
        val_loaders = process_video_path(video_path, model, config, enable_ema, enable_bf16, enable_tile)
        output_root_dir = f"./test_dir/{exp_name}" + ("_continue" if use_continue else "")
        if enable_ema:
            output_root_dir = f"./test_dir/{exp_name}_ema" + ("_continue" if use_continue else "")
        os.makedirs(output_root_dir, exist_ok=True)
        

    for val_idx, val_loader in enumerate(val_loaders):
        val_output_dir = os.path.join(output_root_dir, f"val_dataset_{val_idx}")
        os.makedirs(val_output_dir, exist_ok=True)
        
        print(f"Processing validation dataset {val_idx}")
        

        if enable_ema:
            model.enable_ema()
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch[0], str):  # DualDataLoader格式
                data_type, data = batch
                idx, videos = data  # 解包内部的 [idx, x]
                if data_type == "image":
                    videos = videos.unsqueeze(1)  # 添加时间维度 
            else:  # 原有格式 (idx, x)
                idx, videos = batch
                data_type = None

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            videos = videos.cuda().to(torch.float32)

            
            with torch.no_grad():
                B, T, C, H, W = videos.shape
                T = min(T, 100)  # The model is trained on 49 frames at most
                H_ = H // 16 * 16
                W_ = W // 16 * 16
                T_ = (T-1) // 16 * 16 + 1
                videos = videos[:, :T_, :, :H_, :W_]
                B, T, C, H, W = videos.shape
                print(f"Videos shape: {videos.shape}")
                dtype = torch.bfloat16 # if enable_bf16 else torch.float32
                # 直接处理整个视频，不使用分段
                with torch.autocast(device_type='cuda', dtype=dtype):
                    videos = videos.to(dtype).cuda()
                    with torch.no_grad():
                        print(f"Processing full video shape: {videos.shape}")
                    if hasattr(model, 'forward_unify'):
                        recon, *_ = model.forward_unify(videos, use_continue=use_continue)
                    else:
                        recon, *_ = model.forward(videos)
                    recon = recon.to(torch.float32)

            batch_size = 1
            if recon.dim() < 5:
                recon = recon.view(batch_size, -1, *recon.shape[1:])
            print(f"Recon shape: {recon.shape}")
            # 保存每个样本的视频
            for i, sample_idx in enumerate(idx):
                # 获取原始输入视频和重建视频
                original_video = videos[i]  # 原始输入视频
                recon_video = recon[i]      # 重建输出视频
                
                # 左右拼接输入和输出视频
                # 确保两个视频的帧数相同
                min_frames = min(original_video.shape[0], recon_video.shape[0])
                original_video = original_video[:min_frames]
                recon_video = recon_video[:min_frames]
                
                # 拼接视频 (T, C, H, W) -> (T, C, H, W*2)
                concatenated_video = torch.cat([original_video, recon_video], dim=-1)
                
                output_video_filename = os.path.join(val_output_dir, f"sample_{sample_idx}.mp4")
                save_video(concatenated_video, output_video_filename)



def process_video_path(video_path, model, config, enable_ema, enable_bf16=False, enable_tile=False):
    video_path = Path(video_path)
    if video_path.is_file():
        video_files = [video_path]
    else:
        # 收集所有视频文件
        video_files = []
        video_files.extend(video_path.glob('*.mp4'))
        video_files.extend(video_path.glob('*.avi'))
        video_files.extend(video_path.glob('*.mov'))
        video_files = sorted(video_files)  # 确保顺序一致

    # 创建一个简单的数据加载器类
    class SimpleVideoDataLoader:
        def __init__(self, video_files):
            self.video_files = video_files
            
        def __len__(self):
            return len(self.video_files)
            
        def __iter__(self):
            for idx, video_file in enumerate(self.video_files):
                video_tensor = load_and_preprocess_video(video_file)
                # 确保维度顺序为 (B,T,C,H,W)
                video_tensor = video_tensor.unsqueeze(0)  # 添加batch维度 (1,T,C,H,W)
                yield [idx], video_tensor.contiguous()

    def load_and_preprocess_video(path, target_fps=16):
        cap = cv2.VideoCapture(str(path))
        frames = []
        
        # 获取原始视频的fps
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video fps: {original_fps}")
        
        # 目标长边
        target_long_side = 1280
        target_size = None  # (width, height)
        
        def resize_to_long_side_1080(img):
            nonlocal target_size
            h, w = img.shape[:2]
            if target_size is None:
                long_side = max(h, w)
                scale = target_long_side / float(long_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                target_size = (new_w, new_h)
                print(f"Resize frames from ({h},{w}) -> ({new_h},{new_w}) (long side 1080)")
            # cv2.resize expects (width, height)
            interp = cv2.INTER_AREA if (target_size[0] < w or target_size[1] < h) else cv2.INTER_CUBIC
            return cv2.resize(img, target_size, interpolation=interp)
        
        if target_fps is not None:
            # 计算帧间隔
            frame_interval = original_fps / target_fps
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 根据目标fps采样帧
                if frame_count % max(1, int(frame_interval)) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = resize_to_long_side_1080(frame)
                    frames.append(frame)
                
                frame_count += 1
        else:
            # 原始方式：读取所有帧
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = resize_to_long_side_1080(frame)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from video: {path}")
        
        frames = np.stack(frames)
        # 转换为tensor并进行归一化
        video_tensor = torch.from_numpy(frames).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        video_tensor = video_tensor / 127.5 - 1.0
        
        if target_fps is not None:
            print(f"Loaded {len(frames)} frames at target fps: {target_fps}")
        else:
            print(f"Loaded {len(frames)} frames at original fps: {original_fps}")
            
        return video_tensor


    # 返回数据加载器
    return [SimpleVideoDataLoader(video_files)]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--enable_ema', action='store_true', help='enable ema')
    parser.add_argument('--video_path', type=str, help='path to video file or directory', default=None)
    parser.add_argument('--enable_bf16', action='store_true', help='enable bfloat16 precision')
    parser.add_argument('--enable_tile', action='store_true', help='enable tile mode with overlap')
    parser.add_argument('--segment', action='store_true', help='enable video segmentation (default: disabled)')
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    run_eval(config, args.enable_ema, args.video_path, args.enable_bf16, args.enable_tile, not args.segment)
