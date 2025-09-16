Polynomial Function Approximation with Deep Neural Networks

📌 Overview:

This project explores how a Deep Neural Network (DNN) can approximate a 4th-degree polynomial function using simulated data. The goal was to evaluate how well a simple neural network can capture non-linear relationships and compare its predictions to the true function.

🧮 Problem Setup:
True function:
f(x)=2−5x+0.8x^2−0.3x^3+0.1x^4

Data Generation:
Draw 200 samples of x ~ U[-2, 2]
Generate response y = f(x) + ϵ, where ϵ ~ N(0, 0.4^2)
Store results in two-column dataframe: (x, y)

🔍 Exploratory Data Analysis
Data spans both positive and negative values of x.
Visualization shows:
Black dots: noisy observed data.
Blue curve: true polynomial function.
The curve is U-shaped but skewed due to odd-powered terms.

🏗️ Model Architecture:
Framework: Keras (Sequential API)
Layers:
Input layer with tanh activation.
Hidden layer with 3 tanh neurons.
Output layer with 1 neuron.
Training setup:
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: up to 800
Batch size: 32
20% validation split
Early stopping (patience = 50) to avoid overfitting.

📊 Results
Visualization:
Cyan line: true polynomial.
Red line: neural network predictions.
Black dots: observed noisy samples.

<img width="636" height="760" alt="image" src="https://github.com/user-attachments/assets/ded8d25d-b3c8-4b29-a46b-67721b30ce89" />





