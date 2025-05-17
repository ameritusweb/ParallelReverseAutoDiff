//------------------------------------------------------------------------------
// <copyright file="ILayerConverter.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Linq;

    /// <summary>
    /// Converts to a Layer or a NestedLayersTimeStep.
    /// </summary>
    public class ILayerConverter : JsonConverter
    {
        /// <summary>
        /// Can convert to an ILayer.
        /// </summary>
        /// <param name="objectType">The type of object.</param>
        /// <returns>A value indicating whether it can convert to an ILayer.</returns>
        public override bool CanConvert(Type objectType) => objectType == typeof(ILayer);

        /// <summary>
        /// Read the json.
        /// </summary>
        /// <param name="reader">The JSON reader.</param>
        /// <param name="objectType">The object type.</param>
        /// <param name="existingValue">The existing value.</param>
        /// <param name="serializer">The serializer.</param>
        /// <returns>The object instance.</returns>
        /// <exception cref="JsonSerializationException">Failed to serailzie.</exception>
        public override object ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var jo = JObject.Load(reader);
            var typeProp = jo.Properties()
                 .FirstOrDefault(p => string.Equals(p.Name, "Type", StringComparison.OrdinalIgnoreCase));
            string type = typeProp?.Value?.ToString() ?? string.Empty;

            object retVal = null!;

            switch (type.ToLowerInvariant())
            {
                default:
                case "layer":
                    {
                        var obj = new Layer();
                        serializer.Populate(jo.CreateReader(), obj);
                        retVal = obj;
                        break;
                    }

                case "nested":
                case "nestedlayerstimestep":
                    {
                        var obj = new NestedLayersTimeStep();
                        serializer.Populate(jo.CreateReader(), obj);
                        retVal = obj;
                        break;
                    }

                case "timestep":
                    {
                        var obj = new TimeStep();
                        serializer.Populate(jo.CreateReader(), obj);
                        retVal = obj;
                        break;
                    }
            }

            if (retVal == null)
            {
                throw new InvalidOperationException("Return value is null.");
            }

            return retVal!;
        }

        /// <summary>
        /// Writes to the json.
        /// </summary>
        /// <param name="writer">The json writer.</param>
        /// <param name="value">The object value.</param>
        /// <param name="serializer">The serializer.</param>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            JObject jo = JObject.FromObject(value!, serializer);
            if (value is Layer)
            {
                jo.AddFirst(new JProperty("Type", "Layer"));
            }
            else if (value is NestedLayersTimeStep)
            {
                jo.AddFirst(new JProperty("Type", "NestedLayersTimeStep"));
            }
            else if (value is TimeStep)
            {
                jo.AddFirst(new JProperty("Type", "TimeStep"));
            }

            jo.WriteTo(writer);
        }
    }
}
