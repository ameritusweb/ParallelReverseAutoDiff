using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class SyllableExtractor
    {
        public void Extract()
        {
            // Specify your directory containing JSON files
            string directoryPath = @"E:\\chat";

            // Initialize a HashSet to store unique words
            HashSet<string> uniqueWords = new HashSet<string>();

            var files = Directory.GetFiles(directoryPath, "*.*");

            int i = 0;
            // Iterate over each file in the directory
            foreach (var filePath in files)
            {
                // Read the content of the file
                string content = File.ReadAllText(filePath);

                // Use regex to extract words
                // \w+ matches one or more word characters (letters, digits, or underscores)
                foreach (Match match in Regex.Matches(content, @"[a-zA-Z-]+"))
                {
                    // Add the word to the HashSet
                    uniqueWords.Add(match.Value.ToLower()); // Convert to lower case to ensure case-insensitive uniqueness
                }

                i++;
            }

            HashSet<string> uniqueSyllables = new HashSet<string>();

            // At this point, uniqueWords contains all unique words from all JSON files
            // You can now use this HashSet for further processing
            foreach (var word in uniqueWords)
            {
                var syllables = Syllabifier.Syllabify(word);
                foreach (var syllable in syllables)
                {
                    uniqueSyllables.Add(syllable);
                }
            }
        }
    }
}
