//------------------------------------------------------------------------------
// <copyright file="ArtifactDictionaryGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Newtonsoft.Json;

    /// <summary>
    /// An artifact dictionary generator.
    /// </summary>
    public class ArtifactDictionaryGenerator
    {
        /// <summary>
        /// Generate an artifact dictionary.
        /// </summary>
        public void Generate()
        {
            string[] artifacts = new[]
            {
                "wq", "wr", "wb", "wn", "wp", "wk", "bq", "br", "bb", "bn", "bp", "bk",
                "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
                "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
                "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8",
                "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
                "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
                "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8",
                "o-o", "o-o-o", "e.p.", "#", "$", "+", "=", "=q", "=r", "=b", "=n",
                "w", "b", "k", "q", "-",
            };

            Dictionary<string, int> artifactMap = new Dictionary<string, int>();
            for (int i = 0; i < artifacts.Length; i++)
            {
                artifactMap.Add(artifacts[i], i);
            }

            // Serialize the dictionary to JSON
            string json = JsonConvert.SerializeObject(artifactMap, Formatting.Indented);

            // Define the file path to save the JSON data
            string path1 = "artifacts.json";

            // Save the JSON data to the file
            File.WriteAllText(path1, json);
        }
    }
}
