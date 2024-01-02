using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition
{
    class RandomPairGenerator
    {
        public static List<Tuple<int, int>> GenerateRandomPairs(int numberOfFiles)
        {
            List<int> indices = new List<int>();
            for (int i = 0; i < numberOfFiles; i++)
            {
                indices.Add(i);
            }

            // Shuffle the indices
            Random rng = new Random(Guid.NewGuid().GetHashCode());
            int n = indices.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                int value = indices[k];
                indices[k] = indices[n];
                indices[n] = value;
            }

            // Create pairs from the shuffled indices
            List<Tuple<int, int>> pairs = new List<Tuple<int, int>>();
            for (int i = 0; i < indices.Count - 1; i += 2)
            {
                pairs.Add(new Tuple<int, int>(indices[i], indices[i + 1]));
            }

            // If odd number of files, the last file won't be paired
            if (numberOfFiles % 2 != 0)
            {
                Console.WriteLine("Warning: Odd number of files. Last file will not be paired.");
            }

            return pairs;
        }
    }
}
