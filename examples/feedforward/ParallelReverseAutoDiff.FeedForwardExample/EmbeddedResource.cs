//------------------------------------------------------------------------------
// <copyright file="EmbeddedResource.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using System.Reflection;

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

            throw new Exception("Could not read JSON file");
        }
    }
}
