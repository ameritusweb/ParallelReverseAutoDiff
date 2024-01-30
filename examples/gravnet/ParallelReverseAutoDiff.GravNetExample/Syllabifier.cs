using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public static class Syllabifier
    {
        private static readonly HashSet<char> Vowels = new HashSet<char> { 'a', 'e', 'i', 'o', 'u', 'y' };
        private static readonly List<string> Clusters = new List<string> { "tio", "sio", "cio" };
        private static readonly List<string> ClusterSuffixes = new List<string> { "age" };
        private static readonly List<string> SeparableVowels = new List<string> { "ia", "io", "ua", "uo" };
        private static readonly Dictionary<string, string> CustomMappings = new Dictionary<string, string> {
            { "coope", "CoOpe" },
            { "reen", "ReEn" },
        };

        public static List<string> Syllabify(string word)
        {
            word = word.ToLower(CultureInfo.InvariantCulture);
            string vowels = word;

            foreach (var vowel in Vowels)
            {
                vowels = vowels.Replace(vowel.ToString(), "_");
            }

            foreach (var separableVowel in SeparableVowels)
            {
                word = word.Replace(separableVowel, separableVowel.ToUpper(CultureInfo.InvariantCulture));
            }

            foreach (var cluster in Clusters)
            {
                word = word.Replace($"{cluster.Substring(0, 1)}{cluster.Substring(1).ToUpper(CultureInfo.InvariantCulture)}", cluster);
                word = word.Replace(cluster, $"{cluster.Substring(0, 1).ToUpper(CultureInfo.InvariantCulture)}{cluster.Substring(1)}");
            }

            foreach (var mapping in CustomMappings)
            {
                word = word.Replace(mapping.Key, mapping.Value);
            }

            List<string> syllables = new List<string>();
            StringBuilder currSyllable = new StringBuilder();

            int i = 0;
            while (i < word.Length)
            {
                char ch = word[i];

                bool foundSuffix = false;
                foreach (var suffix in ClusterSuffixes)
                {
                    if (word.Substring(i) == suffix)
                    {
                        currSyllable.Append(word.Substring(i));
                        i += suffix.Length;
                        foundSuffix = true;
                        break;
                    }
                }

                if (foundSuffix)
                {
                    break;
                }

                if (ch.ToString() == ch.ToString().ToUpper())
                {
                    if (!string.IsNullOrEmpty(currSyllable.ToString()))
                    {
                        syllables.Add(currSyllable.ToString().ToLower(CultureInfo.InvariantCulture));
                    }

                    currSyllable.Clear();
                    currSyllable.Append(word[i]);
                    i++;
                }
                else if (Vowels.Contains(ch) || (ch == 'y' && i != 0))
                {
                    currSyllable.Append(ch);
                    i++;
                }
                else
                {
                    currSyllable.Append(word[i]);
                    i++;
                    if (i - 2 >= 0 && i - 2 < vowels.Length && vowels[i - 2] == '_')
                    {
                        syllables.Add(currSyllable.ToString().ToLower(CultureInfo.InvariantCulture));
                        currSyllable.Clear();
                    }
                }
            }

            var remaining = currSyllable.ToString();
            // Add any remaining syllable at the end of the word
            if (!string.IsNullOrEmpty(remaining))
            {
                if (syllables.Any() && remaining.Length == 1 && vowels[vowels.Length - 1] != '_')
                {
                    syllables[syllables.Count - 1] += remaining.ToLower(CultureInfo.InvariantCulture);
                }
                else
                {
                    syllables.Add(remaining);
                }
            }

            return syllables;
        }
    }
}
