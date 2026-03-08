"""Minimal offline inference demo for Qwen3-Omni (text-only input/output)."""
# sudo update-alternatives --set cuda /usr/local/cuda-13.0

from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images
    images[0].save("coffee.png")