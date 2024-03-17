import subprocess
import argparse

def run_script(model):
    script = "flops.py"
    if model == "UNETR" or model == "SwinUNet" or model == "UNet" :
        prefix = "/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/MonAIModel/"
    elif model == "SegTransVAE" or model == "TransBTS" :
        prefix = "/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/3DUnet_Like/"
    sctipts = prefix + script
    command = f"python3 {sctipts} --model {model}"
    output = subprocess.check_output(command, shell=True)
    return output.decode("utf-8")

def main(model):
    output = run_script(model)
    path = f"/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/FLOPS/{model}.txt"
    with open(path, "w") as file:
        file.write(output)
    print("Output has been saved to:", path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    args = parser.parse_args()
    main(args.model)

