using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class Syllabifier
    {
        private static readonly HashSet<char> Vowels = new HashSet<char> { 'a', 'e', 'i', 'o', 'u', 'y' };
        private static readonly List<string> Clusters = new List<string> { "tio", "sio", "cio" };
        private static readonly List<string> ClusterSuffixes = new List<string> { "age" };
        private static readonly List<string> SeparableVowels = new List<string> { "ia", "io", "ua", "uo" };
        private static readonly Dictionary<string, string> CustomMappings = new Dictionary<string, string>() {
            { "coope", "CoOpe" },
            { "reen", "ReEn" },
        };

        public static List<string> Syllabify(string word)
        {
            word = word.ToLower();
            string vowels = word;

            foreach (var vowel in Vowels)
            {
                vowels = vowels.Replace(vowel.ToString(), "_");
            }

            foreach (var separableVowel in SeparableVowels)
            {
                word = word.Replace(separableVowel, separableVowel.ToUpper());
            }

            foreach (var cluster in Clusters)
            {
                word = word.Replace($"{cluster.Substring(0, 1)}{cluster.Substring(1).ToUpper()}", cluster);
                word = word.Replace(cluster, $"{cluster.Substring(0, 1).ToUpper()}{cluster.Substring(1)}");
            }

            foreach (var mapping in CustomMappings)
            {
                word = word.Replace(mapping.Key, mapping.Value);
            }

            List<string> syllables = new List<string>();
            string currentSyllable = "";

            int i = 0;
            while (i < word.Length)
            {
                char ch = word[i];

                bool foundSuffix = false;
                foreach (var suffix in ClusterSuffixes)
                {
                    if (word.Substring(i) == suffix)
                    {
                        currentSyllable += word.Substring(i);
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
                    if (!string.IsNullOrEmpty(currentSyllable))
                    {
                        syllables.Add(currentSyllable.ToLower());
                    }

                    currentSyllable = "";
                    currentSyllable += word[i];
                    i++;
                }
                else if (Vowels.Contains(ch) || (ch == 'y' && i != 0))
                {
                    currentSyllable += ch;
                    i++;
                }
                else
                {
                    currentSyllable += word[i];
                    i++;
                    if (i - 2 >= 0 &&i - 2 < vowels.Length && vowels[i - 2] == '_')
                    {
                        syllables.Add(currentSyllable.ToLower());
                        currentSyllable = "";
                    }
                }
            }

            // Add any remaining syllable at the end of the word
            if (!string.IsNullOrEmpty(currentSyllable))
            {
                if (syllables.Any() && currentSyllable.Length == 1 && vowels[vowels.Length - 1] != '_')
                {
                    syllables[syllables.Count - 1] += currentSyllable.ToLower();
                }
                else
                {
                    syllables.Add(currentSyllable);
                }
            }

            return syllables;
        }

        //public static List<string> SimpleSyllabify(string word)
        //{
        //    List<string> syllables = new List<string>();
        //    string currentSyllable = "";
        //    bool foundVowel = false;

        //    int i = 0;
        //    while (i < word.Length)
        //    {
        //        char ch = word[i];

        //        if (Vowels.Contains(ch) || (ch == 'y' && i != 0))
        //        {
        //            currentSyllable += ch;
        //            i++;

        //            // Continue adding vowels to the current syllable
        //            while (i < word.Length && Vowels.Contains(word[i]))
        //            {
        //                currentSyllable += word[i];
        //                i++;
        //            }

        //            foundVowel = true;
        //        }
        //        else  // Handling consonants
        //        {
        //            bool clusterFound = false;
        //            foreach (var cluster in ConsonantClusters)
        //            {
        //                if (i + cluster.Length <= word.Length && word.Substring(i, cluster.Length).Equals(cluster, StringComparison.OrdinalIgnoreCase))
        //                {
        //                    currentSyllable += cluster;  // Add the consonant cluster to the current syllable
        //                    i += cluster.Length;
        //                    clusterFound = true;
        //                    break;  // Exit the loop once a cluster is found and added
        //                }
        //            }

        //            if (!clusterFound && i < word.Length)
        //            {
        //                // Add individual consonants to the syllable if no cluster is found
        //                currentSyllable += word[i];
        //                i++;
        //            }
        //        }

        //        // If the next character is a vowel or we've reached the end of the word, conclude the current syllable
        //        if (foundVowel && i < word.Length && (Vowels.Contains(word[i]) || i == word.Length) && !string.IsNullOrEmpty(currentSyllable))
        //        {
        //            syllables.Add(currentSyllable);
        //            currentSyllable = "";  // Reset for the next syllable
        //        }
        //    }

        //    // Add any remaining syllable at the end of the word
        //    if (!string.IsNullOrEmpty(currentSyllable))
        //    {
        //        syllables.Add(currentSyllable);
        //    }

        //    return syllables;
        //}
    }
}
