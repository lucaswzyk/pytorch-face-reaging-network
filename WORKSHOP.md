# Age Transformation Workshop

Welcome to the **Age Transformation Workshop**! This guide will help you set up everything you need to get started quickly.

## Prerequisites

Since we're using **GitHub Codespaces**, all necessary system tools like Python are already preinstalled. Let's jump right in!

---

## Setup Instructions

### 1. Install Dependencies  
Run the following command to install all required Python libraries:

```bash
pip install -r requirements.txt
```

Run the following command to install system requirements:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

### 2. Download the Model  
Instead of downloading to your local machine, use this command to download the model directly into your Codespace:

```bash
wget -O models/large-aging-model.h5 https://huggingface.co/spaces/penpen/age-transformation/resolve/main/large-aging-model.h5?download=true
wget -O example-videos/founder_medium_big_cropped.mov https://drive.usercontent.google.com/u/2/uc?id=1x3X7GklCVwCYZvlCZ1R0D3m5i5t7HBjj&export=download
```

This will save the model file (`large-aging-model.h5`) in the current working directory of your Codespace.

---

## You're All Set!

Once the dependencies are installed and the model is downloaded, you're ready to dive into the code and explore age transformation models 🎉.

Happy coding!