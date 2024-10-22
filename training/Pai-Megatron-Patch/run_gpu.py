import torch
import time

def maximize_gpu_utilization():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    # Select GPU device
    device = torch.device("cuda:0")

    # Create a small tensor
    data = torch.ones(10000, 10000, device=device)

    start_time = time.time()

    # Perform matrix multiplication in a loop to keep the GPU busy
    while True:
        for _ in range(10000):
            result = torch.matmul(data, data)

        # Flush CUDA to ensure all tasks are completed
        # torch.cuda.synchronize()

        # elapsed_time = time.time() - start_time
        # if elapsed_time > 60:  # run for 60 seconds
        #     break

    print("Completed intensive computation on GPU.")

if __name__ == "__main__":
    maximize_gpu_utilization()

