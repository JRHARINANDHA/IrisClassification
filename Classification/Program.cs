using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.CSV.Segregate;
using Encog.App.Analyst.CSV.Shuffle;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Persist;
using Encog.Util.CSV;
using Encog.Util.Simple;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Classification
{
    class Program
    {
        public static object EncogDirectoryPersistance { get; private set; }
        public static object EncogUtil { get; private set; }

        static void Main(string[] args)
        {
            Console.WriteLine("######Step 1######");
            Step1();
            Console.WriteLine("######Step 2######");
            Step2();
            Console.WriteLine("######Step 3######");
            Step3();
            Console.WriteLine("######Step 4######");
            Step4();
            Console.WriteLine("######Step 5######");
            Step5();
            Console.WriteLine("######Step 6######");
            Step6();
            Console.WriteLine("Press any key to exit");
            Console.ReadLine();
        }

        #region "Step1:Shuffle"

        static void Step1()
        {
            Console.WriteLine("Shuffle");
            Shuffle(config.BaseFile);
        }

        static void Shuffle(FileInfo source)
        {
            var shuffle = new ShuffleCSV();
            shuffle.Analyze(source, true, CSVFormat.English);
            shuffle.ProduceOutputHeaders = true;
            shuffle.Process(config.ShuffledBaseFile);
        }

        #endregion

        #region "Step2:Segregate"

        static void Step2()
        {
            Console.WriteLine("Segregate");
            Segregate(config.ShuffledBaseFile);
        }

        static void Segregate(FileInfo source)
        {
            var seg = new SegregateCSV();
            seg.Targets.Add(new SegregateTargetPercent(config.TrainingFile, 75));
            seg.Targets.Add(new SegregateTargetPercent(config.TestFile, 25));
            seg.ProduceOutputHeaders = true;
            seg.Analyze(source, true, CSVFormat.English);
            seg.Process();
        }

        #endregion

        #region "Step3:Normalize"

        static void Step3()
        {
            Console.WriteLine("Normalize");
            Normalize();
        }

        static void Normalize()
        {
            var analyst = new EncogAnalyst();

            var wizard = new AnalystWizard(analyst);
            wizard.Wizard(config.BaseFile, true, AnalystFileFormat.DecpntComma);

            var norm = new AnalystNormalizeCSV();
            norm.Analyze(config.TrainingFile, true, CSVFormat.English, analyst);
            norm.ProduceOutputHeaders = true;
            norm.Normalize(config.NormalizedTrainingFile);

            norm.Analyze(config.TestFile, true, CSVFormat.English, analyst);
            norm.Normalize(config.NormlalizedTestFile);

            analyst.Save(config.AnalystFile);
        }

        #endregion

        #region "Step4:CreateNetwork"

        static void Step4()
        {
            Console.WriteLine("CreateNetwork");
            CreateNetwork(config.TrainedNetworkFile);
        }

        static void CreateNetwork(FileInfo networkFile)
        {
            var network = new BasicNetwork();

            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 12));
            network.AddLayer(new BasicLayer(new ActivationTANH(), true, 6));
            network.AddLayer(new BasicLayer(new ActivationTANH(), false, 2));
            network.Structure.FinalizeStructure();
            network.Reset();
            EncogDirectoryPersistence.SaveObject(networkFile, (BasicNetwork)network);
        }

        #endregion

        #region "Step5:Train"

        static void Step5()
        {
            Console.WriteLine("TrainNetwork");
            TrainNetwork();
        }

        static void TrainNetwork()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(config.TrainedNetworkFile);
            var trainingSet = EncogUtility.LoadCSV2Memory(config.NormalizedTrainingFile.ToString(), network.InputCount, network.OutputCount, true, CSVFormat.English, false);

            var train = new ResilientPropagation(network, trainingSet);
            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine("Iteration {0} , Error {1}", epoch, train.Error);
                epoch++;

            } while (train.Error > 0.3);

            EncogDirectoryPersistence.SaveObject(config.TrainedNetworkFile, (BasicNetwork)network);
        }

        #endregion

        #region "Step6:Evaluate"

        static void Step6()
        {
            Console.WriteLine("Evaluation");
            Evaluate();
        }

        static void Evaluate()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(config.TrainedNetworkFile);

            var analyst = new EncogAnalyst();
            analyst.Load(config.AnalystFile.ToString());
            var evaluationSet = EncogUtility.LoadCSV2Memory(config.NormlalizedTestFile.ToString(),network.InputCount,network.OutputCount,true,CSVFormat.English,false);
            int count = 0;
            int correct = 0;

            foreach(var item in evaluationSet)
            {
                count++;
                var output = network.Compute(item.Input);

                var sepal_l = analyst.Script.Normalize.NormalizedFields[0].DeNormalize(item.Input[0]);
                var sepal_w = analyst.Script.Normalize.NormalizedFields[1].DeNormalize(item.Input[1]);
                var petal_l = analyst.Script.Normalize.NormalizedFields[2].DeNormalize(item.Input[2]);
                var petal_w = analyst.Script.Normalize.NormalizedFields[3].DeNormalize(item.Input[3]);
                


                var classCount = (int)analyst.Script.Normalize.NormalizedFields[4].Classes.Count;
                double NormalizedHigh = analyst.Script.Normalize.NormalizedFields[4].NormalizedHigh;
                double NormalizedLow = analyst.Script.Normalize.NormalizedFields[4].NormalizedLow;

                var eq = new Encog.MathUtil.Equilateral(classCount, NormalizedHigh, NormalizedLow);
                var predictedClassInt = eq.Decode(output);
                var predictedClass = analyst.Script.Normalize.NormalizedFields[4].Classes[predictedClassInt].Name;
                var idealClassInt = eq.Decode(item.Ideal);
                var idealClass = analyst.Script.Normalize.NormalizedFields[4].Classes[idealClassInt].Name;

                if(predictedClassInt == idealClassInt)
                {
                    correct++;
                }
                

            }
            Console.WriteLine("Total Test Count {0}", count);
            Console.WriteLine("Correct Count {0}", correct);
            Console.WriteLine("{0} % Success", ((correct * 100) / count));
        }

        #endregion
    }
}
