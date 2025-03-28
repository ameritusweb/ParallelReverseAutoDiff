//------------------------------------------------------------------------------
// <copyright file="ShapeTensorGenerator.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Numerics;
    using Newtonsoft.Json;

    /// <summary>
    /// Generates tensor representations of shapes defined by ShapeConfig objects.
    /// The tensors represent vector fields that can be used in vector-based neural networks.
    /// </summary>
    public class ShapeTensorGenerator
    {
        private readonly int gridSize;
        private readonly float scale;
        private readonly float centerOffset;
        private readonly float normalVectorLength;

        /// <summary>
        /// Initializes a new instance of the <see cref="ShapeTensorGenerator"/> class.
        /// </summary>
        /// <param name="gridSize">Size of the grid for the output tensor (gridSize x gridSize).</param>
        /// <param name="scale">Scale factor for the shape.</param>
        /// <param name="centerOffset">Offset of the shape center from the origin.</param>
        /// <param name="normalVectorLength">Length of normal vectors generated in the tensor.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when gridSize or scale is not positive.
        /// </exception>
        public ShapeTensorGenerator(
            int gridSize,
            float scale = 1.0f,
            float? centerOffset = null,
            float normalVectorLength = 15.0f)
        {
            if (gridSize <= 0)
            {
                throw new ArgumentException("Grid size must be positive");
            }

            if (scale <= 0)
            {
                throw new ArgumentException("Scale must be positive");
            }

            this.gridSize = gridSize;
            this.scale = scale;
            this.centerOffset = centerOffset ?? (gridSize * scale / 2);
            this.normalVectorLength = normalVectorLength;
        }

        /// <summary>
        /// Loads shape configurations from a JSON string using Newtonsoft.Json.
        /// </summary>
        /// <param name="json">The JSON string containing shape configurations.</param>
        /// <returns>A ShapeConfigs object containing all the deserialized configurations.</returns>
        public static ShapeConfigs LoadFromJson(string json)
        {
            var configs = JsonConvert.DeserializeObject<ShapeConfigs>(json);
            configs!.Validate();
            return configs;
        }

        /// <summary>
        /// Generates a 2D tensor representation of a shape defined by the provided configuration.
        /// </summary>
        /// <param name="config">The shape configuration to use.</param>
        /// <param name="rotation">Rotation angle in radians to apply to the shape.</param>
        /// <returns>A 2D array of Vector2 objects representing normal vectors at each grid point.</returns>
        public Vector2?[,] GenerateTensor(ShapeConfig config, float rotation)
        {
            config.Validate();

            var tensor = new Vector2?[this.gridSize, this.gridSize];
            float cellSize = this.scale;
            Vector2 center = new Vector2(this.centerOffset, this.centerOffset);

            foreach (var segment in config.Segments)
            {
                int segmentPoints = (int)((segment.EndAngle - segment.StartAngle) *
                    config.NumPoints / (2 * Math.PI));
                float angleStep = (segment.EndAngle - segment.StartAngle) / segmentPoints;

                for (int i = 0; i <= segmentPoints; i++)
                {
                    float baseAngle = segment.StartAngle + (i * angleStep);
                    float angle = baseAngle + rotation;

                    // Calculate outer and inner points
                    float outerR = this.GetRadius(segment.OuterRadius, baseAngle);
                    float innerR = this.GetRadius(segment.InnerRadius, baseAngle);

                    // Calculate positions
                    Vector2 outerPoint = new Vector2(
                        (float)Math.Cos(angle) * outerR,
                        (float)Math.Sin(angle) * outerR) + center;

                    Vector2 innerPoint = new Vector2(
                        (float)Math.Cos(angle) * innerR,
                        (float)Math.Sin(angle) * innerR) + center;

                    // Calculate normal vectors
                    Vector2 baseNormal = new Vector2(
                        (float)Math.Cos(baseAngle + (Math.PI / 2)),
                        (float)Math.Sin(baseAngle + (Math.PI / 2))) * this.normalVectorLength;
                    Vector2 normalVector = baseNormal.Rotate(rotation);

                    // Map to tensor grid
                    this.MapPointToTensor(outerPoint, normalVector, tensor, cellSize);
                    this.MapPointToTensor(innerPoint, -normalVector, tensor, cellSize);
                }
            }

            return tensor;
        }

        /// <summary>
        /// Calculates the radius for a given angle based on the radius configuration.
        /// Includes variation if specified in the configuration.
        /// </summary>
        /// <param name="config">The radius configuration to use.</param>
        /// <param name="angle">The angle in radians at which to calculate the radius.</param>
        /// <returns>The calculated radius value.</returns>
        private float GetRadius(RadiusConfig config, float angle)
        {
            if (config.Variation == null)
            {
                return config.Base * this.scale;
            }

            return (config.Base + ((float)Math.Sin(angle * config.Variation.Frequency) * config.Variation.Amplitude)) * this.scale;
        }

        /// <summary>
        /// Maps a point and its associated vector to the appropriate cell in the tensor.
        /// </summary>
        /// <param name="point">The point to map in world coordinates.</param>
        /// <param name="vector">The vector to store at the mapped point.</param>
        /// <param name="tensor">The tensor to map into.</param>
        /// <param name="cellSize">The size of each cell in the tensor grid.</param>
        private void MapPointToTensor(Vector2 point, Vector2 vector, Vector2?[,] tensor, float cellSize)
        {
            int gridX = (int)(point.X / cellSize);
            int gridY = (int)(point.Y / cellSize);

            if (gridX >= 0 && gridX < this.gridSize && gridY >= 0 && gridY < this.gridSize)
            {
                tensor[gridY, gridX] = vector;
            }
        }
    }
}