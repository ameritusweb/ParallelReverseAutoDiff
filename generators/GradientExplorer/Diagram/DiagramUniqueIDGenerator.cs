using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Diagram
{
    public class DiagramUniqueIDGenerator : IDisposable
    {
        private static readonly Lazy<DiagramUniqueIDGenerator> LazyLoadedInstance =
            new Lazy<DiagramUniqueIDGenerator>(() => new DiagramUniqueIDGenerator(), true);

        private bool disposed = false;  // To detect redundant Dispose calls

        private long currentID = 0;
        private const long initialPrefix = 1000000000000000000;  // Start with 1 followed by 18 zeros
        private long primaryPrefix = initialPrefix;
        private long secondaryPrefix = initialPrefix;
        private SpinLock spinLock = new SpinLock();
        private volatile int delay = 10000;
        private readonly int delayIncrement = 10000;
        private readonly int maxDelay = int.MaxValue - 10000;

        private readonly ConcurrentQueue<string> idQueue = new ConcurrentQueue<string>();
        private const int batchSize = 1000;
        private const int miniBatchSize = 10;
        private Task generationTask;
        private volatile CancellationTokenSource cancellationTokenSource;

        private const string initialLetters = "AAAAAAAAAAAAAAAAAA"; // 18 As
        private const string lastLetters = "zzzzzzzzzzzzzzzzzz"; // 18 zs
        private readonly string letterset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        private readonly int lettersetLength;
        private readonly char[] currentLetterID;
        private int produceMode = 0;

        public static DiagramUniqueIDGenerator Instance => LazyLoadedInstance.Value;

        private DiagramUniqueIDGenerator()
        {
            cancellationTokenSource = new CancellationTokenSource();
            generationTask = Task.Run(() => GenerateIDs(), cancellationTokenSource.Token);
            lettersetLength = letterset.Length;
            currentLetterID = new char[18];
            for (int i = 0; i < 18; i++)
            {
                currentLetterID[i] = 'A';
            }
        }

        private char IncrementLetter(char currentLetter)
        {
            int idx = letterset.IndexOf(currentLetter);
            int newIdx = (idx + 1) % lettersetLength;
            return letterset[newIdx];
        }

        private void IncrementLetterID()
        {
            int carry = 1;
            for (int i = 17; i >= 0; i--)
            {
                if (carry == 0)
                {
                    break;
                }
                char newChar = IncrementLetter(currentLetterID[i]);
                if (newChar == 'A')
                {
                    carry = 1;
                }
                else
                {
                    carry = 0;
                }
                currentLetterID[i] = newChar;
            }
        }

        private void ProduceNextID(int size)
        {
            bool lockTaken = false;
            try
            {
                spinLock.Enter(ref lockTaken);
                for (int i = 0; i < size; ++i)
                {
                    long nextID = Interlocked.Increment(ref currentID);
                    string id = GenerateID(nextID);
                    idQueue.Enqueue(id);
                }
            }
            finally
            {
                if (lockTaken) spinLock.Exit();
            }
        }

        private void GenerateIDs()
        {
            while (true)
            {
                CancellationToken token = cancellationTokenSource.Token;

                if (idQueue.Count < batchSize)
                {
                    Interlocked.Exchange(ref delay, delayIncrement);
                    ProduceNextID(batchSize);
                }
                else if (delay < maxDelay)
                {
                    Interlocked.Add(ref delay, delayIncrement);
                }

                // Sleep with cancellation token
                try
                {
                    Task.Delay(delay, token).Wait();
                }
                catch (TaskCanceledException)
                {
                    cancellationTokenSource.Dispose();
                    cancellationTokenSource = new CancellationTokenSource();
                }
            }
        }

        private string GenerateID(long nextID)
        {
            string id = "";

            // Roll-over logic for currentID and primaryPrefix
            if (nextID >= long.MaxValue)
            {
                Interlocked.Exchange(ref currentID, 0);
                if (primaryPrefix == initialPrefix)
                {
                    Interlocked.Increment(ref produceMode);
                }
                Interlocked.Increment(ref primaryPrefix);

                // Roll-over logic for primaryPrefix and secondaryPrefix
                if (primaryPrefix >= long.MaxValue)
                {
                    Interlocked.Exchange(ref primaryPrefix, initialPrefix);
                    if (secondaryPrefix == initialPrefix)
                    {
                        Interlocked.Increment(ref produceMode);
                    }
                    Interlocked.Increment(ref secondaryPrefix);

                    // If all are maxed out, use a Letter ID
                    if (secondaryPrefix >= long.MaxValue)
                    {
                        Interlocked.Exchange(ref secondaryPrefix, initialPrefix);
                        if (new string(currentLetterID) == initialLetters)
                        {
                            Interlocked.Increment(ref produceMode);
                        }
                        IncrementLetterID();

                        // If all are maxed out again, use GUIDs
                        if (new string(currentLetterID) == lastLetters)
                        {
                            Interlocked.Increment(ref produceMode);
                        }
                    }
                }
            }

            id = ProduceMode(nextID);

            return id;
        }

        private string ProduceMode(long nextID)
        {
            switch (produceMode)
            {
                case 0:
                    return $"{nextID}";
                case 1:
                    return $"{primaryPrefix}{nextID}";
                case 2:
                    return $"{secondaryPrefix}{primaryPrefix}{nextID}";
                case 3:
                    return $"{new string(currentLetterID)}{secondaryPrefix}{primaryPrefix}{nextID}";
                default:
                    return Guid.NewGuid().ToString();
            }
        }

        public string GetNextID()
        {
            if (idQueue.TryDequeue(out string id))
            {
                return id;
            }
            else
            {

                ProduceNextID(miniBatchSize);

                if (idQueue.TryDequeue(out string producedID))
                {
                    return producedID;
                }
                else
                {
                    // Cancel the sleep timer in GenerateIDs()
                    cancellationTokenSource.Cancel();

                    Console.WriteLine("Fallback to GUID.");

                    // Fallback logic
                    return Guid.NewGuid().ToString();
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources.
                    cancellationTokenSource?.Cancel();
                    cancellationTokenSource?.Dispose();
                    cancellationTokenSource = null;

                    // Optionally, wait for the task to complete.
                    generationTask?.Wait();
                    generationTask?.Dispose();
                    generationTask = null;
                }

                // Dispose unmanaged resources here if any.

                disposed = true;
            }
        }

        // Finalizer
        ~DiagramUniqueIDGenerator()
        {
            Dispose(false);
        }
    }
}
