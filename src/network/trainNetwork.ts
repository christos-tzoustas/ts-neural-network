import { loadData } from "./mnistLoader";
import * as fs from "fs";
import * as path from "path";
import { Network } from "./Network";

// Load the data
const { modelTrainingData, modelTestData } = loadData();

console.log(
  `Beginning training with ${modelTrainingData.length} training samples and ${modelTestData.length} test samples`
);

// Initialize the network
const net = new Network([784, 30, 10]);

// Train the network
const learningRate = 3;
const epochs = 30;
const miniBatchSize = 10;

net.applySGD(
  modelTrainingData,
  epochs,
  miniBatchSize,
  learningRate,
  modelTestData
);

// Save the weights and biases to a JSON file
const model = net.exportModel();
const outputDir = path.resolve(__dirname, "../../static");
const outputPath = path.join(outputDir, "network-weights.json");

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(model));

console.log("Model saved to static/network-weights.json");
