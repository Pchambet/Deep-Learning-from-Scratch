# Deep Learning, From First Principles.

<p align="center">
  <img src="banner.png" alt="Deep Learning Journey — by Pierre Chambet" width="800">
</p>

<h1 align="center">Deep Learning Journey</h1>
<p align="center">
  From neuron to CNN, built and explained from scratch.<br>
  <a href="https://www.linkedin.com/in/pierre-chambet-289a5b220/">LinkedIn</a> • 
  <a href="https://github.com/Pchambet">GitHub</a>
</p>

> As an engineering student, I wasn't satisfied with just calling `.fit()`.
>
> I had simple questions that demanded deep answers: "What *is* a neuron?", "How does Backpropagation *actually* work, mathematically?", "Why *this* activation function?".
>
> This repository is the result of that frustration. It's my personal 100+ page journey to build neural networks from the ground up, starting from first principles.
>
> **This isn't 'just another guide'. It's the guide I wish I'd had. And now, it's here.**

---

## 🚀 The 3-Part Journey: From Idea to Network

This project is a complete, 3-part series designed to build a deep, intuitive understanding of neural networks.

### 📚 Part 1: The Foundation – The ANN "From Scratch"

Here, we build a complete Artificial Neural Network (a fully-connected network) using **nothing but Python and NumPy**. We leave PyTorch and TensorFlow at the door to understand what they *really* do for us.

* **The Math:** A full, step-by-step derivation of **Backpropagation**. This is the engine of deep learning, explained without shortcuts.
* **The Code:** A line-by-line implementation of the neuron, the dense layer, activation functions, and the training loop.
* **[Read The Full Guide (PDF) →](Part_1_ANN_from_Scratch/Guide_1_ANN_from_Scratch.pdf)**
* **[View The Companion Notebook →](Part_1_ANN_from_Scratch/ANN_Companion_Notebook.ipynb)**

### 💡 Part 2: The Proof – Our "Scratch" ANN vs. MNIST

Theory is one thing. But does it work? We take our hand-built neural network from Part 1 and unleash it on the "Hello, World!" of Deep Learning: the **MNIST** dataset.

* **The Challenge:** Train our raw NumPy code to recognize handwritten digits.
* **The Result:** A proof-of-concept that our 'from-scratch' code works, complete with performance analysis and visualizations.
* **[View The MNIST Case Study (Notebook) →](Part_2_MNIST_Case_Study/MNIST_ANN_from_Scratch.ipynb)**

### ⚡ Part 3: The Toolkit – Mastering Frameworks with CNNs

Now that we truly *understand* the mechanics, we can master the industry tools. We use **PyTorch** and **TensorFlow** to build a **Convolutional Neural Network (CNN)**—the right tool for the job.

* **The Goal:** Build a high-performance CNN for MNIST and compare the two leading frameworks.
* **The Verdict:** A hands-on analysis of PyTorch (research, flexibility) vs. TensorFlow (production, scalability).
* **[View The PyTorch Notebook →](Part_3_Frameworks_CNN/CNN_with_PyTorch.ipynb)**
* **[View The TensorFlow Notebook →](Part_3_Frameworks_CNN/CNN_with_TensorFlow.ipynb)**

---

## 👋 About This Project

This project was a personal challenge to look under the hood, to prove to myself that I understood the 'why' and not just the 'how'.

If you're a student, developer, or engineer who is also curious, I hope this helps. If you find an error or just want to connect, feel free to open an Issue.

**Build with me.**
