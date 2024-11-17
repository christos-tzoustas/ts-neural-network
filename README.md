
# ts-neural-network

A basic digit recognition neural network implemented in TypeScript.

## Description

This project implements a simple neural network for digit recognition using the MNIST dataset. It also implements a frontend for visualizing the results of testing the model. The network is written in TypeScript and uses TensorFlow.js for matrix computations.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/christos-tzoustas/ts-neural-network.git
cd ts-neural-network
npm install
```

## Usage

### Training the Network

To train the neural network, run:

```bash
npm run train
```

This will train the network on the MNIST dataset and save the trained model to `static/network-weights.json`.

### Running the Frontend

To start the development server and view the frontend, run:

```bash
npm run dev
```

Then, open your browser and navigate to [http://localhost:1234](http://localhost:1234).

## Acknowledgments

This project is based on the code examples from Michael Nielsen's *Neural Networks and Deep Learning* book, translated from Python to TypeScript.
