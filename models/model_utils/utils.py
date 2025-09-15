import torch

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def flatten_state_dict(nested_dict, parent_key="", separator="."):
    """
    将嵌套的字典展开为扁平化的字典，键以 'xxx.XXX.XXX' 格式表示。
    
    :param nested_dict: 嵌套字典
    :param parent_key: 父级 key，用于构建完整路径
    :param separator: 分隔符，默认为 '.'
    :return: 扁平化字典
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_state_dict(value, full_key, separator))
        else:
            flat_dict[full_key] = value
    return flat_dict


def save_video(tensor, filename, fps=4, bitrate='2000k'):
    # tensor shape should be (t, c, h, w), assuming 3 channels (RGB)
    t, c, h, w = tensor.shape
    
    # Ensure the values are in the range [-1, 1] and convert to [0, 255]
    tensor = tensor.clamp(-1, 1)  # Ensure values are in the range [-1, 1]
    tensor = (tensor + 1) * 127.5  # Convert to [0, 255]
    tensor = tensor.permute(0, 2, 3, 1)  # Change to (t, h, w, c) for moviepy
    
    # Convert tensor to uint8 type
    tensor = tensor.to(torch.uint8)
    
    # Convert tensor to a list of frames
    frames = [frame.cpu().numpy() for frame in tensor]
    
    # Create a video clip from the frames
    video_clip = ImageSequenceClip(frames, fps=fps)
    
    # Write the video to file with a specified bitrate 
    video_clip.write_videofile(filename, codec='libx264', bitrate=bitrate)