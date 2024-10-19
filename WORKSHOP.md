Here’s an updated version of your README that reflects the addition of your notebook and guides the user to it:

---

# **Age Transformation Workshop**

Welcome to the **Age Transformation Workshop**! This guide will help you set up everything you need to explore the magic of age transformation using machine learning models.

We've now made things even easier with a **Google Colab notebook**—no need to worry about setting up dependencies manually. If you're interested in trying it out, follow the instructions below!

---

## **Prerequisites**

We recommend running the project in **Google Colab** for a hassle-free experience with GPU acceleration. Alternatively, **GitHub Codespaces** can also be used if you prefer working in a more integrated coding environment.

---

## **How to Use the Google Colab Notebook**

1. **Open the Colab Notebook**  
   Access the notebook here: [Age Transformation Notebook](https://colab.research.google.com/drive/1wIs4Fpr2oB5uPp3EevPhod3lII1AzWvu?usp=sharing)  

2. **Follow the Instructions in the Notebook**  
   The notebook guides you through every step, from installing dependencies to running the video and image processing pipeline.

3. **Use GPU for Faster Inference**  
   In Colab, go to **Runtime > Change Runtime Type** and make sure the **Hardware Accelerator** is set to **GPU** for faster video processing.

---

## **Setup for GitHub Codespaces (Optional)**

If you want to explore the project locally or in **GitHub Codespaces**, follow these steps:  

### 1. **Install Dependencies**  
Run the following command to install the required Python libraries:

```bash
pip install -r requirements.txt
```

Additionally, install system-level dependencies:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

### 2. **Download the Model and Example Video**  
Instead of downloading files locally, use these commands to download the assets directly into your Codespace:

```bash
wget -O models/large-aging-model.h5 https://huggingface.co/spaces/penpen/age-transformation/resolve/main/large-aging-model.h5?download=true
wget -O example-videos/founder_medium_big_cropped.mov https://drive.usercontent.google.com/u/2/uc?id=1x3X7GklCVwCYZvlCZ1R0D3m5i5t7HBjj&export=download
```

---

## **You're All Set!**

Whether you're using the **notebook in Colab** or working in **GitHub Codespaces**, you're now ready to explore age transformation models 🎉.  

- **Want to see your results side-by-side?** Check out the widget in the notebook for easy video comparison.
- **Encounter issues?** Make sure your runtime uses GPU for optimal performance.  

Happy coding and enjoy transforming time! 🚀

---

This version provides a clear path for users, emphasizing the notebook for ease of use while still including instructions for Codespaces.