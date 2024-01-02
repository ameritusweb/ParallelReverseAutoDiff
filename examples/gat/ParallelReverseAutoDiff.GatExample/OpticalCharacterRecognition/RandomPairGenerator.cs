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
            // Step 1: Generate all possible pairs in both orders
            List<Tuple<int, int>> allPairs = new List<Tuple<int, int>>();
            for (int i = 0; i < numberOfFiles; i++)
            {
                for (int j = i + 1; j < numberOfFiles; j++)
                {
                    allPairs.Add(new Tuple<int, int>(i, j));
                    allPairs.Add(new Tuple<int, int>(j, i));
                }
            }

            // Step 2: Shuffle the generated pairs
            Random rng = new Random(Guid.NewGuid().GetHashCode());
            int n = allPairs.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                Tuple<int, int> value = allPairs[k];
                allPairs[k] = allPairs[n];
                allPairs[n] = value;
            }

            return allPairs;
        }
    }
}
