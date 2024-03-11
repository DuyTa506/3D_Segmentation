

# Final project for DIP 2024 Lectures


### Table of contents
1. [Overview](#overview)
2. [Technical Report](#technical-report)
3. [Installation](#installation)
4. [Metric](#metric)
5. [Training](#Training)
6. [Inference](#inference)
7. [Logs and Visualization](#logs)

<a name = "overview" ></a>

<a name = "technical-report" ></a>

<a name = "metric" ></a>

<a name = "installation" ></a>
 - Prerequisites :
   Before take a deep dive into the project, ensure that you'd installed all requirements packages by this command :
   If you have CUDA kernel acceleration, install Pytorch with CUDA
   
   * I recommend the version of Pytorch and CUDA is 1.13 and 11.6, check the [Pytorch Homepage](https://pytorch.org/get-started/previous-versions/) for more details !

   * After that , run this command to install dependecies :
     ```
     pip install -r requirements.txt
     ```

<a name = "Training" ></a>
 - There are two type of model wrapper architecture : SegTransVAE and MONAI wrapper model
   * For SegTransVAE training , move the work directory to .... and run this command :
     
       ```
         python/python3 train.py
       ```
   * For MONAI wrapper model training , move the work directory to .... and run this command :
     
       ```
         python/python3 train.py --model_type ... --batch_size ... --epochs ... --volume ... --test_size ...
       ```
       Where the model_type can be : UNet, Segresnet, SWINUNETR, UNETR and so on. You can modify the code for more choice of model type
     
<a name = "inference" ></a>
<a name = "inference" ></a>
<a name = "logs" ></a>




















---
license: apache-2.0
---
