import argparse
import imageio
import os


def imgs2gif(frames_dir, duration):
    """
    img_dir: directory for inference results
    duration: duration = 1 / fps
    """
    frames = []
    for idx in sorted(os.listdir(frames_dir)):
        img = os.path.join(frames_dir, idx)
        if img.endswith('jpg'):
            frames.append(imageio.imread(img))
    save_name = '.'.join([frames_dir, 'gif'])
    imageio.mimsave(save_name, frames, 'GIF', duration=duration)
    print(save_name, 'saved!')

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--frames_dir', type=str, default='./inference/YOWO_infer/HorseRiding')
    parser.add_argument('--duration', type=float, default=0.04)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    imgs2gif(args.frames_dir, args.duration)