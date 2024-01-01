using ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition;

namespace ParallelReverseAutoDiff.GatExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            OpticalCharacterRecognitionNetworkTrainer trainer = new OpticalCharacterRecognitionNetworkTrainer();
            Task.Run(async () => await trainer.Train()).Wait();
        }
    }
}