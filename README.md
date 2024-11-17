
# ts-neural-network

A basic digit recognition neural network implemented in **TypeScript**, using **TensorFlow.js** and the **MNIST** dataset. This project is designed to train and evaluate a neural network for recognizing handwritten digits from the MNIST dataset.

## Table of Contents

- [ts-neural-network](#ts-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running in Development Mode](#running-in-development-mode)
    - [Building the Project](#building-the-project)
    - [Running Tests](#running-tests)
  - [Neural Network Architecture](#neural-network-architecture)
    - [Key Components:](#key-components)
  - [License](#license)
  - [Example MNIST Usage:](#example-mnist-usage)
  - [Author](#author)

---

## Project Overview

This project implements a simple feedforward neural network from scratch in TypeScript, trained to recognize handwritten digits using the MNIST dataset. It uses **TensorFlow.js** for matrix operations and backpropagation and allows users to train and evaluate the network.

## Features

- **Custom Feedforward Neural Network**: Built from scratch using TensorFlow.js for training and inference.
- **Backpropagation Algorithm**: Implements the backpropagation algorithm to adjust weights and biases using stochastic gradient descent.
- **MNIST Data**: Uses the MNIST dataset for training and testing.
- **Training and Inference**: Train the network and make predictions on new inputs.

## Technologies Used

- **TypeScript**: Main language used for the project.
- **TensorFlow.js**: For tensor operations and implementing the neural network's backpropagation.
- **MNIST**: A library to load the MNIST dataset for training and testing.
- **Node.js**: Runtime environment for executing TypeScript and running the server-side code.

## Installation

To install and set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd ts-neural-network
   ```

2. **Install dependencies**:
   Install both development and production dependencies using `npm`:
   ```bash
   npm install
   ```

## Usage

### Running in Development Mode

To start the project in development mode with **hot reloading**, use the following command:

```bash
npm run dev
```

This command uses **nodemon** to automatically reload the server when changes are detected in the TypeScript files.

### Building the Project

To build the TypeScript code into JavaScript (in the `dist` folder), run:

```bash
npm run build
```

The `build` command uses TypeScript's compiler (`tsc`) to transpile the `.ts` files into `.js`.

### Running Tests

To run the tests with **Jest**, execute:

```bash
npm run test
```

To watch for file changes and re-run tests automatically:

```bash
npm run test:watch
```

## Neural Network Architecture

The neural network implemented in this project is a simple feedforward network with the following structure:

1. **Input Layer**: 784 neurons (28x28 pixels flattened for MNIST images).
2. **Hidden Layer**: 30 neurons.
3. **Output Layer**: 10 neurons (representing digits 0 through 9).

### Key Components:

- **Activation Function**: The network uses the sigmoid activation function.
- **Cost Function**: The cost function used is the mean squared error.
- **Backpropagation**: The network uses backpropagation to compute gradients for weights and biases, enabling the network to learn.
- **Stochastic Gradient Descent (SGD)**: The model is trained using mini-batch stochastic gradient descent to optimize the weights and biases.

## License

This project is licensed under the MIT License.

---

## Example MNIST Usage:

```typescript
import { Network } from './Network';
import { loadData } from './mnistLoader';

// Load the data
const { trainingData, testData } = loadData();

// Initialize the network
const net = new Network([784, 30, 10]);

// Train the network using Stochastic Gradient Descent
const learningRate = 3;
const epochs = 30;
const miniBatchSize = 10;

net.applySGD(trainingData, epochs, miniBatchSize, learningRate, testData);
```

---

## Author

**Christos Tzoustas**
