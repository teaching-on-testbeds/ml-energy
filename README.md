# ml-energy

---

**Attribution**: This sequence of notebooks is closely based on one by Sundeep Rangan, from his [IntroML GitHub repo](https://github.com/sdrangan/introml/). 

---

In this sequence of experiments, you will implement an audio classifier using a neural network-based architecture. Here, you’ll build a model to identify the instrument used to generate the audio. With a focus on building energy-efficient models, once the efficacy of your model has been established, you’ll calculate the energy consumption of the model during training. Furthermore, you’ll observe how the energy consumption varies with changing training hyperparameters across metrics such as TrainToAccuracy (TTA) and EnergyToAccuracy (ETA).

Procedure: First, you'll run the `reserve.ipynb` notebook to bring up a resource on Chameleon and configure it with the software needed to run this experiment. At the end of this notebook, you'll set up an SSH tunnel between your local device and a Jupyter notebook server that you just created on your Chameleon resource. Then, you'll open the notebook server in your local browser and run the sequence of notebooks you see there.