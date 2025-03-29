using ParallelReverseAutoDiff.PRAD.VectorTools;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD.VectorTools
{
    public class ShapeTensorGeneratorTests
    {
        [Fact]
        public void TestShapeTensorGenerator()
        {
            string jsonConfig = @"{
                ""shapes"": {
                    ""C"": {
                        ""numPoints"": 72,
                        ""segments"": [
                            {
                                ""startAngle"": 1.5,
                                ""endAngle"": 6.28,
                                ""outerRadius"": {
                                    ""base"": 80,
                                    ""variation"": {
                                        ""frequency"": 2,
                                        ""amplitude"": 10
                                    }
                                },
                                ""innerRadius"": {
                                    ""base"": 60,
                                    ""variation"": {
                                        ""frequency"": 2,
                                        ""amplitude"": 8
                                    }
                                }
                            }
                        ]
                    },
                    ""O"": {
                        ""numPoints"": 72,
                        ""segments"": [
                            {
                                ""startAngle"": 0,
                                ""endAngle"": 6.28,
                                ""outerRadius"": {
                                    ""base"": 80,
                                    ""variation"": {
                                        ""frequency"": 2,
                                        ""amplitude"": 10
                                    }
                                },
                                ""innerRadius"": {
                                    ""base"": 60,
                                    ""variation"": {
                                        ""frequency"": 2,
                                        ""amplitude"": 8
                                    }
                                }
                            }
                        ]
                    },
                    ""S"": {
                        ""numPoints"": 72,
                        ""segments"": [
                            {
                                ""startAngle"": 0,
                                ""endAngle"": 3.14,
                                ""outerRadius"": {
                                    ""base"": 60,
                                    ""variation"": {
                                        ""frequency"": 1,
                                        ""amplitude"": 5
                                    }
                                },
                                ""innerRadius"": {
                                    ""base"": 40,
                                    ""variation"": {
                                        ""frequency"": 1,
                                        ""amplitude"": 4
                                    }
                                }
                            },
                            {
                                ""startAngle"": 3.14,
                                ""endAngle"": 6.28,
                                ""outerRadius"": {
                                    ""base"": 100,
                                    ""variation"": {
                                        ""frequency"": 1,
                                        ""amplitude"": 5
                                    }
                                },
                                ""innerRadius"": {
                                    ""base"": 80,
                                    ""variation"": {
                                        ""frequency"": 1,
                                        ""amplitude"": 4
                                    }
                                }
                            }
                        ]
                    }
                }
            }";

            try
            {
                var configs = ShapeTensorGenerator.LoadFromJson(jsonConfig);
                var generator = new ShapeTensorGenerator(
                    gridSize: 15,
                    scale: 20.0f,
                    normalVectorLength: 1.0f
                );
                var tensor = generator.GenerateTensor(configs.Shapes["C"], 0.0f);
            }
            catch (Exception e)
            {

            }
        }
    }
}
