import * as tf from "@tensorflow/tfjs";
import * as mnist from "mnist";

/**
 * Load the MNIST data and return it as a tuple containing the training data,
 * and the test data.
 */
function loadData() {
  const data = mnist.set(40000, 10000);

  return {
    trainingData: data.training,
    testData: data.test,
  };
}

/**
 * Return a tuple containing (trainingData, validationData, testData).
 * Based on loadData, but the format is more convenient for use in our
 * implementation of neural networks.
 */
export function loadDataWrapper() {
  const { trainingData, testData } = loadData();

  const trainingInputs = trainingData.map((d) =>
    tf.tensor2d(d.input, [784, 1])
  );
  const trainingResults = trainingData.map((d) =>
    tf.tensor2d(d.output, [10, 1])
  );
  const trainingDataFormatted: Array<[tf.Tensor2D, tf.Tensor2D]> =
    trainingInputs.map((input, i) => [input, trainingResults[i]]);

  const testInputs = testData.map((d) => tf.tensor2d(d.input, [784, 1]));
  const testResults = testData.map((d) => d.output.indexOf(1));
  const testDataFormatted: Array<[tf.Tensor2D, number]> = testInputs.map(
    (input, i) => [input, testResults[i]]
  );

  return {
    trainingData: trainingDataFormatted,
    testData: testDataFormatted,
  };
}
