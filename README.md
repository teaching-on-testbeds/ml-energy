# Considering Energy-to-Accuracy in Training Machine Learning Models

In this sequence of experiments, you will implement a neural network classifier to identify the instrument that generated an audio sample. Then, by calculating the energy used by the GPU during training, you’ll observe how the energy required to train a model to a given validation accuracy varies with training hyperparameters such as learning rate and batch size.

Procedure: First, you'll run the `reserve.ipynb` notebook to bring up a resource on Chameleon and configure it with the software needed to run this experiment. At the end of this notebook, you'll set up an SSH tunnel between your local device and a Jupyter notebook server that you just created on your Chameleon resource. Then, you'll open the notebook server in your local browser and run the sequence of notebooks you see there.

---

**Attribution**: The audio classification material is based on an assignment by Sundeep Rangan, from his [IntroML GitHub repo](https://github.com/sdrangan/introml/). 

This material is based upon work supported by the National Science Foundation under Grant No. 2230079 and Grant No. 2226408.

---

