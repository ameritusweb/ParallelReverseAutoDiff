//------------------------------------------------------------------------------
// <copyright file="AdjacencyMatrixGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;
    using Newtonsoft.Json;

    /// <summary>
    /// An adjacency matrix generator.
    /// </summary>
    public class AdjacencyMatrixGenerator
    {
        /// <summary>
        /// Generate an adjacency matrix.
        /// </summary>
        public void Generate()
        {
            var dir = Directory.GetCurrentDirectory();
            string filePath = dir + "\\paths.txt";
            List<string> lines = File.ReadAllLines(filePath).ToList();

            // List to hold paths
            List<List<Position>> paths = new List<List<Position>>();

            foreach (var line in lines)
            {
                List<Position> path = new List<Position>();
                for (int i = 0; i < line.Length; i += 2)
                {
                    Position pos = new Position(line.Substring(i, 2));
                    path.Add(pos);
                }

                // Add path to paths list
                paths.Add(path);
            }

            // Create an adjacency matrix
            int[][] adjacencyMatrix = new int[paths.Count][];

            for (int i = 0; i < paths.Count; i++)
            {
                adjacencyMatrix[i] = new int[paths.Count];

                for (int j = 0; j < paths.Count; j++)
                {
                    if (i == j)
                    {
                        adjacencyMatrix[i][j] = 0; // No self-loops
                    }
                    else
                    {
                        // Paths are connected if they share a position
                        adjacencyMatrix[i][j] = paths[i].Intersect(paths[j]).Any() ? 1 : 0;
                    }
                }
            }

            // Create a dictionary to hold the DOK representation
            Dictionary<Tuple<int, int>, int> dokMatrix = new Dictionary<Tuple<int, int>, int>();

            // Convert the adjacency matrix to DOK
            for (int i = 0; i < paths.Count; i++)
            {
                for (int j = 0; j < paths.Count; j++)
                {
                    if (adjacencyMatrix[i][j] != 0)
                    {
                        // Add the non-zero values to the dictionary
                        dokMatrix[new Tuple<int, int>(i, j)] = adjacencyMatrix[i][j];
                    }
                }
            }

            // The adjacencyMatrix now represents the connections between paths.
            // Serialize the nested dictionary to JSON
            string json = JsonConvert.SerializeObject(dokMatrix, Formatting.Indented);

            // Define the file path to save the JSON data
            string path1 = "adjacency.json";

            // Save the JSON data to the file
            File.WriteAllText(path1, json);
        }
    }
}
