import argparse
import time
from PIL import Image
import torch
from tqdm import trange
import torchvision.transforms as transforms

def size_of_tensor_in_bytes(encoding):
    return encoding.nelement() * encoding.element_size()

def create_batch(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess_pipeline = transforms.Compose(
         [
         transforms.Resize(256), 
         transforms.CenterCrop(256), 
         transforms.ToTensor(), 
         normalize
         ]
    )
    img = Image.open('leopard_frog_s_001876.png')

    if img.mode == "L":
        img = img.convert("RGB")  

    if preprocess_pipeline is not None:
        img = preprocess_pipeline(img)

    imgs =[]    
    for i in range(0,batch_size):  
        imgs.append(img)

    return torch.stack(imgs)

def stress_vram_transfer(
        batch_size=512,
        warmup=5,
        repeats=100,
        frame_shape=(3, 3840, 2160),
        use_pinned_memory=True,
):
    tensor = create_batch(256)
    #tensor = torch.randn((batch_size, *frame_shape))


    print(f"Batch Size(Mb): {size_of_tensor_in_bytes(tensor)/1024/1024}")

    if use_pinned_memory:
        tensor = tensor.pin_memory()

    for device_id in range(torch.cuda.device_count()):
        print(f"Starting test for device {device_id}: {torch.cuda.get_device_properties(device_id)}")
        for _ in trange(warmup, desc="warmup"):
            tensor = tensor.to(device=device_id, non_blocking=use_pinned_memory)
            tensor = tensor.to(device="cpu", non_blocking=use_pinned_memory)
            if use_pinned_memory:
                torch.cuda.current_stream(device=device_id).synchronize()
        start = time.perf_counter()
        for _ in trange(repeats, desc="test"):
            tensor = tensor.to(device=device_id, non_blocking=use_pinned_memory)
            #tensor = tensor.to(device="cpu", non_blocking=use_pinned_memory)
            if use_pinned_memory:
                torch.cuda.current_stream(device=device_id).synchronize()
        end = time.perf_counter()
        print(f"Total time taken: {end-start:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--frame_shape", type=int, nargs=3, default=(3, 3840, 2160))
    parser.add_argument("--use_pinned_memory", type=bool, default=True)
    parser.add_argument('--no_pin', dest='use_pinned_memory', action='store_false')
    args = parser.parse_args()

    args = dict(vars(args))
    print(args)
    stress_vram_transfer(**args)