

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Numerics;
    using Newtonsoft.Json;

    public class ShapeTensorGenerator
    {
        private readonly int gridSize;
        private readonly float scale;
        private readonly float centerOffset;
        private readonly float normalVectorLength;

        public ShapeTensorGenerator(
            int gridSize,
            float scale = 1.0f,
            float? centerOffset = null,
            float normalVectorLength = 15.0f)
        {
            if (gridSize <= 0)
                throw new ArgumentException("Grid size must be positive");
            if (scale <= 0)
                throw new ArgumentException("Scale must be positive");

            this.gridSize = gridSize;
            this.scale = scale;
            this.centerOffset = centerOffset ?? (gridSize * scale / 2);
            this.normalVectorLength = normalVectorLength;
        }

        private float GetRadius(RadiusConfig config, float angle)
        {
            if (config.Variation == null)
                return config.Base * scale;

            return (config.Base +
                (float)Math.Sin(angle * config.Variation.Frequency) *
                config.Variation.Amplitude) * scale;
        }

        public Vector2?[,] GenerateTensor(ShapeConfig config, float rotation)
        {
            config.Validate();

            var tensor = new Vector2?[gridSize, gridSize];
            float cellSize = scale;
            Vector2 center = new Vector2(centerOffset, centerOffset);

            foreach (var segment in config.Segments)
            {
                int segmentPoints = (int)((segment.EndAngle - segment.StartAngle) *
                    config.NumPoints / (2 * Math.PI));
                float angleStep = (segment.EndAngle - segment.StartAngle) / segmentPoints;

                for (int i = 0; i <= segmentPoints; i++)
                {
                    float baseAngle = segment.StartAngle + i * angleStep;
                    float angle = baseAngle + rotation;

                    // Calculate outer and inner points
                    float outerR = GetRadius(segment.OuterRadius, baseAngle);
                    float innerR = GetRadius(segment.InnerRadius, baseAngle);

                    // Calculate positions
                    Vector2 outerPoint = new Vector2(
                        (float)Math.Cos(angle) * outerR,
                        (float)Math.Sin(angle) * outerR
                    ) + center;

                    Vector2 innerPoint = new Vector2(
                        (float)Math.Cos(angle) * innerR,
                        (float)Math.Sin(angle) * innerR
                    ) + center;

                    // Calculate normal vectors
                    Vector2 baseNormal = new Vector2(
                        (float)Math.Cos(baseAngle + Math.PI / 2),
                        (float)Math.Sin(baseAngle + Math.PI / 2)
                    ) * normalVectorLength;
                    Vector2 normalVector = baseNormal.Rotate(rotation);

                    // Map to tensor grid
                    MapPointToTensor(outerPoint, normalVector, tensor, cellSize);
                    MapPointToTensor(innerPoint, -normalVector, tensor, cellSize);
                }
            }

            return tensor;
        }

        private void MapPointToTensor(Vector2 point, Vector2 vector,
            Vector2?[,] tensor, float cellSize)
        {
            int gridX = (int)(point.X / cellSize);
            int gridY = (int)(point.Y / cellSize);

            if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize)
            {
                tensor[gridY, gridX] = vector;
            }
        }

        public static ShapeConfigs LoadFromJson(string json)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            var configs = JsonSerializer.Deserialize<ShapeConfigs>(json, options);
            configs.Validate();
            return configs;
        }
    }
}
