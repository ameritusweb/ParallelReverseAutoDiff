using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.FeedForward
{
    /// <summary>
    /// Helper class to read embedded resources.
    /// </summary>
    public static class EmbeddedResource
    {
        /// <summary>
        /// Reads a JSON file from the embedded resources.
        /// </summary>
        /// <param name="name">The namespace.</param>
        /// <param name="file">The file to read.</param>
        /// <returns>The text of the file.</returns>
        public static string ReadAllJson(string name, string file)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = $"{name}.{file}.json";

            using (Stream? stream = assembly.GetManifestResourceStream(resourceName))
            {
                if (stream != null)
                {
                    using (StreamReader reader = new StreamReader(stream))
                    {
                        string result = reader.ReadToEnd();
                        return result;
                    }
                }
            }

            throw new InvalidOperationException("Could not read JSON file");
        }
    }
}
