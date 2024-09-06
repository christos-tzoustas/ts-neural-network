import * as tf from "@tensorflow/tfjs";
import * as mnist from "mnist";

/**
 * Load the MNIST data and return it as a tuple containing the training data,
 * the validation data, and the test data.
 */
function loadData() {
  const data = mnist.set(8000, 2000);

  const validationDataSize = 1000;
  const validationData = data.test.slice(0, validationDataSize);
  const testData = data.test.slice(validationDataSize);

  return {
    trainingData: data.training,
    validationData,
    testData,
  };
}

// Creates a 10-dimensional one-hot encoded vector for the given number 'j'
// async function vectorizedResult(j: number): Promise<tf.Tensor2D> {
//   // Create a 10x1 tensor filled with zeros
//   const e = tf.zeros<tf.Rank.R2>([10, 1]);

//   // Convert the tensor to a TensorBuffer so that we can modify individual elements
//   const buffer = await e.buffer();

//   // Set the value at index j to 1 (one-hot encoding)
//   buffer.set(1, j, 0);

//   // Return the modified buffer as a tensor again
//   return buffer.toTensor();
// }

/**
 * Return a tuple containing (trainingData, validationData, testData).
 * Based on loadData, but the format is more convenient for use in our
 * implementation of neural networks.
 */
export function loadDataWrapper() {
  const { trainingData, validationData, testData } = loadData();

  const trainingInputs = trainingData.map((d) =>
    tf.tensor2d(d.input, [784, 1])
  );
  const trainingResults = trainingData.map((d) =>
    tf.tensor2d(d.output, [10, 1])
  );
  const trainingDataFormatted: Array<[tf.Tensor2D, tf.Tensor2D]> =
    trainingInputs.map((input, i) => [input, trainingResults[i]]);

  const validationInputs = validationData.map((d) =>
    tf.tensor2d(d.input, [784, 1])
  );
  // is this the way to get the number out?
  const validationResults = validationData.map((d) => d.output.indexOf(1));
  const validationDataFormatted: Array<[tf.Tensor2D, number]> =
    validationInputs.map((input, i) => [input, validationResults[i]]);

  const testInputs = testData.map((d) => tf.tensor2d(d.input, [784, 1]));
  const testResults = testData.map((d) => d.output.indexOf(1));
  const testDataFormatted: Array<[tf.Tensor2D, number]> = testInputs.map(
    (input, i) => [input, testResults[i]]
  );

  return {
    trainingData: trainingDataFormatted,
    validationData: validationDataFormatted,
    testData: testDataFormatted,
  };
}
