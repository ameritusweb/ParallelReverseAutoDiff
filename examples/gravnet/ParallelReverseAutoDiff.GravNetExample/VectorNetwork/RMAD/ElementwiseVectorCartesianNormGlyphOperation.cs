//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianNormGlyphOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.GravNetExample.GlyphNetwork;
    using ParallelReverseAutoDiff.GravNetExample.VectorFieldNetwork;
    using ParallelReverseAutoDiff.GravNetExample.VectorNetwork;

    /// <summary>
    /// Element-wise cartesian glyph operation.
    /// </summary>
    public class ElementwiseVectorCartesianNormGlyphOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;
        private CalculatedValues[,] calculatedValues;
        private double[] dNormX_dX;
        private double[] dNormX_dY;
        private double[] dNormY_dX;
        private double[] dNormY_dY;
        private VectorNetwork vectorNetwork;
        private readonly VectorFieldNetwork vectorFieldNetwork;
        private readonly GlyphNetwork glyphNetwork;

        public ElementwiseVectorCartesianNormGlyphOperation(VectorNetwork vectorNetwork)
        {
            this.vectorNetwork = vectorNetwork;
        }

        public ElementwiseVectorCartesianNormGlyphOperation(VectorFieldNetwork vectorFieldNetwork)
        {
            this.vectorFieldNetwork = vectorFieldNetwork;
        }

        public ElementwiseVectorCartesianNormGlyphOperation(GlyphNetwork glyphNetwork)
        {
            this.glyphNetwork = glyphNetwork;
        }

        public ElementwiseVectorCartesianNormGlyphOperation(NeuralNetwork net)
        {

        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            if (net is VectorNetwork vectorNetwork)
            {
                return new ElementwiseVectorCartesianNormGlyphOperation(vectorNetwork);
            } else if (net is VectorFieldNetwork vectorFieldNetwork)
            {
                return new ElementwiseVectorCartesianNormGlyphOperation(vectorFieldNetwork);
            } else if (net is GlyphNetwork glyphNetwork)
            {
                return new ElementwiseVectorCartesianNormGlyphOperation(glyphNetwork);
            }
            else
            {
                return new ElementwiseVectorCartesianNormGlyphOperation(net);
            }
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector glyph function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector glyph operation.</param>
        /// <param name="input2">The second input to the element-wise vector glyph operation.</param>
        /// <param name="weights">The weights.</param>
        /// <returns>The output of the element-wise vector glyph operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            int sectionCount = 225; // Total sections
            int rows = input1.Rows;
            int cols = input1.Cols / 2; // Assuming cols are split between magnitudes and angles

            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.calculatedValues = new CalculatedValues[rows, cols];

            // Initialize output structure for section sums
            this.Output = new Matrix(sectionCount, 2); // 225 sections, each with a sum X and Y

            // Example structure to hold section sums, initialized to zero
            double[] sectionSumsX = new double[sectionCount];
            double[] sectionSumsY = new double[sectionCount];

            this.dNormX_dX = new double[sectionCount];
            this.dNormX_dY = new double[sectionCount];
            this.dNormY_dX = new double[sectionCount];
            this.dNormY_dY = new double[sectionCount];

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    // Compute vector components as before
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + cols];
                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + cols];

                    double x1 = magnitude * Math.Cos(angle);
                    double y1 = magnitude * Math.Sin(angle);
                    double x2 = wMagnitude * Math.Cos(wAngle);
                    double y2 = wMagnitude * Math.Sin(wAngle);

                    double sumX = x1 + x2;
                    double sumY = y1 + y2;

                    // a function that determines the section index for this vector
                    int sectionIndex = DetermineSectionIndex(i, j);

                    // Compute resultant vector magnitude and angle
                    double resultMagnitude = Math.Sqrt((sumX * sumX) + (sumY * sumY)) * weights[i, j];
                    double resultAngle = Math.Atan2(sumY, sumX);

                    double weightedSumX = resultMagnitude * Math.Cos(resultAngle);
                    double weightedSumY = resultMagnitude * Math.Sin(resultAngle);

                    // Apply weights and sum into the correct section
                    sectionSumsX[sectionIndex] += weightedSumX;
                    sectionSumsY[sectionIndex] += weightedSumY;

                    // Backpropagation
                    double dSumX_dAngle = -magnitude * Math.Sin(angle);
                    double dSumX_dWAngle = -wMagnitude * Math.Sin(wAngle);
                    double dSumY_dAngle = magnitude * Math.Cos(angle);
                    double dSumY_dWAngle = wMagnitude * Math.Cos(wAngle);
                    double dSumX_dMagnitude = Math.Cos(angle);
                    double dSumX_dWMagnitude = Math.Cos(wAngle);
                    double dSumY_dMagnitude = Math.Sin(angle);
                    double dSumY_dWMagnitude = Math.Sin(wAngle);

                    double dResultMagnitude_dSumX = sumX / Math.Sqrt((sumX * sumX) + (sumY * sumY)) * weights[i, j];
                    double dResultMagnitude_dSumY = sumY / Math.Sqrt((sumX * sumX) + (sumY * sumY)) * weights[i, j];
                    double dResultAngle_dSumX = -sumY / ((sumX * sumX) + (sumY * sumY));
                    double dResultAngle_dSumY = sumX / ((sumX * sumX) + (sumY * sumY));
                    double dResultMagnitude_dWeight = Math.Sqrt((sumX * sumX) + (sumY * sumY));

                    double dWeightedSumX_dResultMagnitude = Math.Cos(resultAngle);
                    double dWeightedSumX_dResultAngle = -resultMagnitude * Math.Sin(resultAngle);
                    double dWeightedSumY_dResultMagnitude = Math.Sin(resultAngle);
                    double dWeightedSumY_dResultAngle = resultMagnitude * Math.Cos(resultAngle);

                    double dSectionSumsX_dWeight = dWeightedSumX_dResultMagnitude * dResultMagnitude_dWeight;
                    this.calculatedValues[i, j].DSectionSumsX_dWeight = dSectionSumsX_dWeight;
                    double dSectionSumsY_dWeight = dWeightedSumY_dResultMagnitude * dResultMagnitude_dWeight;
                    this.calculatedValues[i, j].DSectionSumsY_dWeight = dSectionSumsY_dWeight;

                    double dSectionSumsX_dMagnitude = dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dMagnitude
                                                      +
                                                      dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dMagnitude
                                                      + 
                                                      dWeightedSumX_dResultAngle * dResultAngle_dSumX * dSumX_dMagnitude
                                                      +
                                                      dWeightedSumX_dResultAngle * dResultAngle_dSumY * dSumY_dMagnitude;
                    this.calculatedValues[i, j].DSectionSumsX_dMagnitude = dSectionSumsX_dMagnitude;

                    double dSectionSumsX_dAngle = dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dAngle
                                                 +
                                                 dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dAngle
                                                 +
                                                 dWeightedSumX_dResultAngle * dResultAngle_dSumX * dSumX_dAngle
                                                 +
                                                 dWeightedSumX_dResultAngle * dResultAngle_dSumY * dSumY_dAngle;
                    this.calculatedValues[i, j].DSectionSumsX_dAngle = dSectionSumsX_dAngle;

                    double dSectionSumsX_dWMagnitude = dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dWMagnitude
                                                     +
                                                     dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dWMagnitude
                                                     +
                                                     dWeightedSumX_dResultAngle * dResultAngle_dSumX * dSumX_dWMagnitude
                                                     +
                                                     dWeightedSumX_dResultAngle * dResultAngle_dSumY * dSumY_dWMagnitude;
                    this.calculatedValues[i, j].DSectionSumsX_dWMagnitude = dSectionSumsX_dWMagnitude;

                    double dSectionSumsX_dWAngle = dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dWAngle
                                                +
                                                dWeightedSumX_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dWAngle
                                                +
                                                dWeightedSumX_dResultAngle * dResultAngle_dSumX * dSumX_dWAngle
                                                +
                                                dWeightedSumX_dResultAngle * dResultAngle_dSumY * dSumY_dWAngle;
                    this.calculatedValues[i, j].DSectionSumsX_dWAngle = dSectionSumsX_dWAngle;

                    double dSectionSumsY_dMagnitude = dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dMagnitude
                                                      +
                                                      dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dMagnitude
                                                      +
                                                      dWeightedSumY_dResultAngle * dResultAngle_dSumX * dSumX_dMagnitude
                                                      +
                                                      dWeightedSumY_dResultAngle * dResultAngle_dSumY * dSumY_dMagnitude;
                    this.calculatedValues[i, j].DSectionSumsY_dMagnitude = dSectionSumsY_dMagnitude;

                    double dSectionSumsY_dAngle = dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dAngle
                                                 +
                                                 dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dAngle
                                                 +
                                                 dWeightedSumY_dResultAngle * dResultAngle_dSumX * dSumX_dAngle
                                                 +
                                                 dWeightedSumY_dResultAngle * dResultAngle_dSumY * dSumY_dAngle;
                    this.calculatedValues[i, j].DSectionSumsY_dAngle = dSectionSumsY_dAngle;

                    double dSectionSumsY_dWMagnitude = dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dWMagnitude
                                                     +
                                                     dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dWMagnitude
                                                     +
                                                     dWeightedSumY_dResultAngle * dResultAngle_dSumX * dSumX_dWMagnitude
                                                     +
                                                     dWeightedSumY_dResultAngle * dResultAngle_dSumY * dSumY_dWMagnitude;
                    this.calculatedValues[i, j].DSectionSumsY_dWMagnitude = dSectionSumsY_dWMagnitude;

                    double dSectionSumsY_dWAngle = dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumX * dSumX_dWAngle
                                                +
                                                dWeightedSumY_dResultMagnitude * dResultMagnitude_dSumY * dSumY_dWAngle
                                                +
                                                dWeightedSumY_dResultAngle * dResultAngle_dSumX * dSumX_dWAngle
                                                +
                                                dWeightedSumY_dResultAngle * dResultAngle_dSumY * dSumY_dWAngle;
                    this.calculatedValues[i, j].DSectionSumsY_dWAngle = dSectionSumsY_dWAngle;
                }
            });

            // Compile section sums into the output matrix
            for (int k = 0; k < sectionCount; k++)
            {
                var magnitude = Math.Sqrt((sectionSumsX[k] * sectionSumsX[k]) + (sectionSumsY[k] * sectionSumsY[k]));
                this.dNormX_dX[k] = (sectionSumsY[k] * sectionSumsY[k]) / (magnitude * magnitude * magnitude);
                this.dNormY_dY[k] = (sectionSumsX[k] * sectionSumsX[k]) / (magnitude * magnitude * magnitude);
                this.dNormX_dY[k] = -sectionSumsX[k] * sectionSumsY[k] / (magnitude * magnitude * magnitude);
                this.dNormY_dX[k] = this.dNormX_dY[k];
                this.Output[k, 0] = sectionSumsX[k] / magnitude;
                this.Output[k, 1] = sectionSumsY[k] / magnitude;
            }

            return Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int sectionCount = 225; // Total sections

            Matrix dInput1 = new Matrix(input1.Rows, input1.Cols);
            Matrix dInput2 = new Matrix(input2.Rows, input2.Cols);
            Matrix dWeights = new Matrix(weights.Rows, weights.Cols);

            // Iterate over the sections to compute gradients of section sums from dOutput
            for (int k = 0; k < sectionCount; k++)
            {
                // For each section, dOutput provides the gradient w.r.t. the normalized output vector's magnitude and angle
                // These need to be related back to the unnormalized section sums (dSectionSumsX and dSectionSumsY)
                double dNormX = dOutput[k, 0]; // Gradient w.r.t normalized X component of the output
                double dNormY = dOutput[k, 1]; // Gradient w.r.t normalized Y component of the output

                double nX = (dNormX * this.dNormX_dX[k]) + (dNormY * this.dNormY_dX[k]);
                double nY = (dNormX * this.dNormX_dY[k]) + (dNormY * this.dNormY_dY[k]);

                // Updating gradients with respect to resultMagnitude and resultAngle
                Parallel.For(0, this.input1.Rows, i =>
                {
                    for (int j = 0; j < this.input1.Cols / 2; j++)
                    {
                        var values = this.calculatedValues[i, j];

                        // Update dWeights with direct contributions from glyphX and glyphY
                        dWeights[i, j] += nX * values.DSectionSumsX_dWeight;
                        dWeights[i, j] += nY * values.DSectionSumsY_dWeight;

                        // Apply chain rule to propagate back to dInput1 and dInput2
                        dInput1[i, j] += nX * values.DSectionSumsX_dMagnitude;
                        dInput1[i, j] += nY * values.DSectionSumsY_dMagnitude;

                        dInput1[i, j + (this.input1.Cols / 2)] += nX * values.DSectionSumsX_dAngle;
                        dInput1[i, j + (this.input1.Cols / 2)] += nY * values.DSectionSumsY_dAngle;

                        dInput2[i, j] += nX * values.DSectionSumsX_dWMagnitude;
                        dInput2[i, j] += nY * values.DSectionSumsY_dWMagnitude;

                        dInput2[i, j + (this.input2.Cols / 2)] += nX * values.DSectionSumsX_dWAngle;
                        dInput2[i, j + (this.input2.Cols / 2)] += nY * values.DSectionSumsY_dWAngle;
                    }
                });
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

        private int DetermineSectionIndex(int row, int col)
        {
            // Define section sizes and total sections per row and column
            const int rowsPerRegularSection = 34;
            const int colsPerRegularSection = 25;
            const int totalSectionsPerCol = 15; // Given by the problem statement

            // Determine the section row and column based on the input row and column
            int sectionRow = row / rowsPerRegularSection;
            int sectionCol = col / colsPerRegularSection;

            // Adjust for the special-sized sections
            if (row >= 14 * rowsPerRegularSection)
            {
                sectionRow = 14; // Last row section
            }

            if (col >= 14 * colsPerRegularSection)
            {
                sectionCol = 14; // Last column section
            }

            // Calculate the flat section index
            int sectionIndex = (sectionRow * totalSectionsPerCol) + sectionCol;

            return sectionIndex;
        }

        private struct CalculatedValues
        {
            public double DSectionSumsX_dMagnitude { get; internal set; }
            public double DSectionSumsX_dAngle { get; internal set; }
            public double DSectionSumsX_dWMagnitude { get; internal set; }
            public double DSectionSumsX_dWAngle { get; internal set; }
            public double DSectionSumsY_dMagnitude { get; internal set; }
            public double DSectionSumsY_dAngle { get; internal set; }
            public double DSectionSumsY_dWMagnitude { get; internal set; }
            public double DSectionSumsY_dWAngle { get; internal set; }
            public double DSectionSumsX_dWeight { get; internal set; }
            public double DSectionSumsY_dWeight { get; internal set; }
        }
    }
}
