namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;

    // Define the abstract base class for all operations.
    public abstract class Operation : IOperation
    {

        // Constructor
        public Operation()
        {
            // Initialize properties
            Inputs = new List<string>();
            Outputs = new List<string>();
            BackwardAdjacentOperations = new List<IOperation>();
            BackwardDependencyCounts = new List<int>();
            AccumulatedGradients = new List<(double[][]?, double[][]?)>();
            Tasks = new List<Task>();
            BackwardDependencies = new List<List<string>>();
            VisitedFrom = new List<string>();
        }

        public void Reset()
        {
            //Debug.WriteLine($"Resetting {SpecificId}");
            VisitedCount = 0;
            VisitedFrom?.Clear();
            AccumulatedGradients.Clear();
            SyncSemaphore?.Dispose();
            SyncSemaphore = null;
            IsComplete = false;
            Tasks?.Clear();
        }

        public bool IsComplete { get; set; }

        // The time step of the current operation
        public int TimeStepIndex { get; set; } = -1;

        // The layer index of the current operation
        public int LayerIndex { get; set; } = -1;

        // The type of operation (e.g. MatrixMultiplyOperation, MatrixAddOperation)
        public Type OperationType { get; set; }

        // Returns true if there's a next operation in the sequence
        public bool HasNext
        {
            get
            {
                return Next != null;
            }
        }

        // Reference to the next operation in the sequence
        public IOperation Next { get; set; }

        // Operation ID
        public string Id { get; set; }

        // Specific ID of the operation
        public string SpecificId { get; set; }

        // Private field to store the output of the operation
        protected double[][] _output;

        // Returns the output of the operation
        public virtual double[][] GetOutput()
        {
            return _output;
        }

        // Abstract method to perform backward pass, must be implemented by derived classes
        public abstract (double[][]?, double[][]?) Backward(double[][] dOutput);

        // Property to store the gradient destination objects
        public object[] GradientDestinations { get; set; }

        // Send the calculated gradient to the appropriate destination object
        public virtual void AccumulateGradient((double[][]?, double[][]?) dOutput)
        {
            var array3D = MatrixUtils.Reassemble(dOutput).ToList();
            for (int k = array3D.Count; k < GradientDestinations.Length; ++k)
            {
                array3D.Add(dOutput.Item2);
            }
            if (GradientDestinations != null && GradientDestinations.Length > 0)
            {
                for (int d = 0; d < GradientDestinations.Length; ++d)
                {
                    var gradientResultTo = (double[][])GradientDestinations[d];
                    if (gradientResultTo != null)
                    {
                        var output = array3D[d];
                        int numRows = gradientResultTo.Length;
                        int numCols = gradientResultTo[0].Length;
                        for (int i = 0; i < numRows; ++i)
                        {
                            for (int j = 0; j < numCols; ++j)
                            {
                                gradientResultTo[i][j] += output[i][j];
                            }
                        }
                    }
                }
            }
        }

        // Property to store the name of the result variable
        public string ResultToName { get; set; }

        // Copies the result of the operation to the specified destination
        public virtual void ResultTo(Func<int, int, object> func)
        {
            var oo = func(TimeStepIndex, LayerIndex);
            double[][] o = null;
            if (oo is Operation)
            {
                o = ((Operation)oo).GetOutput();
            }
            else
            {
                o = (double[][])oo;
            }
            int numRows = _output.Length;
            int numCols = _output[0].Length;
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    o[i][j] = _output[i][j];
                }
            }
        }

        // Initialize the operation with the specified starting point index
        public virtual void Initialize(int startingPointIndex)
        {
            OutputDependencyCount = BackwardDependencyCounts[startingPointIndex];
            InitializeLock();
            InitializeSyncSemaphore();
        }

        // Initialize the lock object
        public virtual void InitializeLock()
        {
            if (Lock == null)
            {
                Lock = new ReaderWriterLockSlim();
            }
        }

        // Initialize the synchronization semaphore
        public virtual void InitializeSyncSemaphore()
        {
            if (SyncSemaphore == null)
            {
                SyncSemaphore = new SemaphoreSlim(0, OutputDependencyCount);
            }
        }

        // The parameters to the Forward function for this operation
        public object[] Parameters { get; set; }

        // The backward tasks running for this operation
        public List<Task> Tasks { get; set; }

        // The specific ID of the operations whose outputs are the inputs to the Forward function for this operation
        public List<string> Inputs { get; set; }

        // The specific ID of the operations who take in this operation's output as input
        public List<string> Outputs { get; set; }

        // The input to the Backward function for this operation
        public object BackwardInput { get; set; }

        // The backward dependencies for this operation
        public List<List<string>> BackwardDependencies { get; set; }

        // Which node this node was visited from
        public List<string> VisitedFrom { get; set; }

        // The operations that are next when traversing the computational graph via the backward pass
        public List<IOperation> BackwardAdjacentOperations { get; set; }

        // The number of operations that take this operation's output as input based on the timestep that you start at when doing the backward pass
        public List<int> BackwardDependencyCounts { get; set; }

        // The accumulated gradients from all output dependent operations
        public List<(double[][]?, double[][]?)> AccumulatedGradients { get; set; }

        // The accumulated gradients from all backward passes through this operation node
        public (double[][]?, double[][]?) CalculatedGradient { get; set; }

        // For the current backward pass, the number of operations that take this operation's output as input
        public int OutputDependencyCount { get; set; }

        // The number of times this operation node has been visited during a specific pass
        public int VisitedCount { get; set; }

        // A lock to handle issues that arise from concurrent access to shared resources
        public ReaderWriterLockSlim Lock { get; set; }

        // A semaphore to synchronize visitor instances to make sure nodes aren't passed through multiple times during a pass through the computational graph
        public SemaphoreSlim SyncSemaphore { get; set; }
    }
}
