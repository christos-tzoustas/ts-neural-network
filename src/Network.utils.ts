import * as tf from "@tensorflow/tfjs";

/**
 * Sigmoid function applied element-wise on the input tensor.
 * @param z - Input tensor.
 * @returns - Tensor with the sigmoid function applied element-wise.
 */
export function sigmoid<T extends tf.Tensor>(z: T): T {
  return tf.div(1, tf.add(1, tf.exp(tf.neg(z))));
}

// Derivative of the sigmoid function
export function sigmoidPrime<T extends tf.Tensor>(z: T): T {
  const sigmoidZ = sigmoid(z);
  return tf.mul(sigmoidZ, tf.sub(1, sigmoidZ));
}

// Shuffles the array in-place using Fisher-Yates algorithm.
export function shuffle<T extends tf.Tensor>(array: [T, T][]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}
