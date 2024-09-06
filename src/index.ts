import { Network } from "./Network";
import { loadDataWrapper } from "./mnistLoader";

// load data
const { trainingData, testData } = loadDataWrapper();

// init network
const networkLayerSizes = [784, 30, 10];
const net = new Network(networkLayerSizes);

// train network
const learningRate = 3;
const epochs = 30;
const miniBatchSize = 10;

net.applySGD(trainingData, epochs, miniBatchSize, learningRate, testData);
