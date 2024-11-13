using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.Json;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Vision
{
    public class HSLImageLoader
    {
        private static JsonSerializerOptions GetJsonOptions()
        {
            return new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                NumberHandling = JsonNumberHandling.AllowReadingFromString
            };
        }

        public static async Task<HSLImageData> LoadFromFileAsync(string filePath)
        {
            try
            {
                string jsonString = await File.ReadAllTextAsync(filePath);
                return JsonSerializer.Deserialize<HSLImageData>(jsonString, GetJsonOptions());
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load HSL data from {filePath}", ex);
            }
        }

        public static HSLImageData LoadFromFile(string filePath)
        {
            try
            {
                string jsonString = File.ReadAllText(filePath);
                return JsonSerializer.Deserialize<HSLImageData>(jsonString, GetJsonOptions());
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load HSL data from {filePath}", ex);
            }
        }
    }
}
