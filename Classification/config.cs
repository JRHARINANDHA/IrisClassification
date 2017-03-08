using Encog.Util.File;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Classification
{
   public static class config
    {
        public static FileInfo BasePath = new FileInfo(@"E:\Project\ML\CarData\");

        #region "Step1"

        public static FileInfo BaseFile = FileUtil.CombinePath(BasePath, "Car_Data.csv");
        public static FileInfo ShuffledBaseFile = FileUtil.CombinePath(BasePath, "Car_ShuffledData.csv");

        #endregion

        #region "Step2"

        public static FileInfo TrainingFile = FileUtil.CombinePath(BasePath, "Car_Training.csv");
        public static FileInfo TestFile = FileUtil.CombinePath(BasePath, "Car_Test.csv");

        #endregion

        #region "Step3"

        public static FileInfo NormalizedTrainingFile = FileUtil.CombinePath(BasePath, "Car_Training_Norm.csv");
        public static FileInfo NormlalizedTestFile = FileUtil.CombinePath(BasePath, "Car_Test_Norm.csv");
        public static FileInfo AnalystFile = FileUtil.CombinePath(BasePath, "Car_Analyst.ega");

        #endregion

        #region "Step4"

        public static FileInfo TrainedNetworkFile = FileUtil.CombinePath(BasePath, "Car_trained_network.eg");

        #endregion
    }
}
