//------------------------------------------------------------------------------
// <copyright file="FourDimensionalMatrixJsonConverter.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Linq;

    /// <summary>
    /// A JSON converter for a FourDimensionalMatrix.
    /// </summary>
    public class FourDimensionalMatrixJsonConverter : JsonConverter
    {
        /// <summary>
        /// Can convert.
        /// </summary>
        /// <param name="objectType">The object type.</param>
        /// <returns>A value indicating whether it can convert.</returns>
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(FourDimensionalMatrix);
        }

        /// <summary>
        /// Write JSON.
        /// </summary>
        /// <param name="writer">The writer.</param>
        /// <param name="value">The value.</param>
        /// <param name="serializer">The serializer.</param>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            FourDimensionalMatrix matrix = value as FourDimensionalMatrix ?? throw new InvalidOperationException("FourDimensionalMatrix cannot be null.");
            writer.WriteStartObject();
            writer.WritePropertyName("UniqueId");
            serializer.Serialize(writer, matrix.UniqueId);
            writer.WritePropertyName("DeepMatrixArrayValues");
            serializer.Serialize(writer, matrix.DeepMatrixArrayValues);
            writer.WriteEndObject();
        }

        /// <summary>
        /// Read JSON.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="objectType">The object type.</param>
        /// <param name="existingValue">The existing value.</param>
        /// <param name="serializer">The serializer.</param>
        /// <returns>The JSON.</returns>
        public override object ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            JObject item = JObject.Load(reader);
            int? uniqueId = item["UniqueId"]?.ToObject<int>();
            DeepMatrix[]? deepMatrixArrayValues = item["DeepMatrixArrayValues"]?.ToObject<DeepMatrix[]>();
            return new FourDimensionalMatrix(uniqueId!.Value, deepMatrixArrayValues!);
        }
    }
}
