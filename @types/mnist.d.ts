/* eslint-disable no-unused-vars */
type GrowToSize<T, N extends number, A extends T[]> = A["length"] extends N
  ? A
  : GrowToSize<T, N, [...A, T]>;

export type FixedArray<T, N extends number> = GrowToSize<T, N, []>;

declare module "mnist" {
  type DataSet = {
    input: FixedArray<number, 784>;
    output: FixedArray<number, 10>;
  };

  type MnistData = {
    training: Array<DataSet>;
    test: Array<DataSet>;
  };

  export function set(trainingAmount: number, testAmount: number): MnistData;
}
