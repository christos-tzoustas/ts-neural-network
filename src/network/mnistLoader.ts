import * as tf from "@tensorflow/tfjs";
import * as mnist from "mnist";

export function loadData() {
  const { training, test } = mnist.set(7500, 1000);

  const trainingInputTensors = training.map((d) =>
    tf.tensor2d(d.input, [784, 1])
  );
  const trainingOutputTensors = training.map((d) =>
    tf.tensor2d(d.output, [10, 1])
  );
  const modelTrainingData: Array<[tf.Tensor2D, tf.Tensor2D]> =
    trainingInputTensors.map((input, i) => [input, trainingOutputTensors[i]]);

  const testInputTensors = test.map((d) => tf.tensor2d(d.input, [784, 1]));
  const testOutputTensors = test.map((d) => d.output.indexOf(1));
  const modelTestData: Array<[tf.Tensor2D, number]> = testInputTensors.map(
    (input, i) => [input, testOutputTensors[i]]
  );

  return {
    mnistTrainingData: training,
    mnistTestData: test,
    modelTrainingData,
    modelTestData,
  };
}
