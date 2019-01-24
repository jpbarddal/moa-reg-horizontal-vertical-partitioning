package moa.experiments;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.HDDM_A_Test;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.classifiers.meta.AddExpContinuous;
import moa.classifiers.meta.RandomSubspacesRegression;
import moa.classifiers.meta.SFNRegressor;
import moa.classifiers.rules.AMRulesRegressor;
import moa.classifiers.social.SFNRegressor2;
import moa.classifiers.trees.FIMTDD;
import moa.classifiers.trees.ORTO;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.TimingUtils;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.WindowRegressionPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.streams.ExampleStream;

import java.io.*;
import java.util.HashMap;


// java -Xms8g -Xmx8g -cp moa-pom.jar -javaagent:sizeofag.jar moa.experiments.RegressionExperiments > log.log 2> error.log
public class RegressionExperiments {

    private static final int INSTANCES_EVAL = 100;
    private static final String PATH_ARFFS = "./datasets/";
    private static final String PATH_RESULTS = "./results/";
    private static final int TYPE_BASIC = 0;
    private static final int TYPE_WINDOWED = 1;

    public static void main(String args[]) throws IOException {
        HashMap<String, Regressor> hsLearners = getLearners();
        HashMap<String, ArffFileStream> hsStreams = getStreams();

        prepareOutputFolder();

        // runs the experiments
        System.out.println("Starting experiments...");
        for(String strExperiment : hsStreams.keySet()){
            for(String strLearner : hsLearners.keySet()){
                System.out.println("Running... " + strExperiment + "\t" + strLearner);
                //prepares the learner
                Regressor reg = hsLearners.get(strLearner);
                ((Classifier) reg).resetLearning();
                ((Classifier) reg).prepareForUse();

                // prepares the stream
                ExampleStream stream = hsStreams.get(strExperiment);
                ((ArffFileStream) stream).prepareForUse();
                stream.restart();

                // runs the experiment
                runExperiment(strExperiment, strLearner, stream, reg);
            }
        }
        System.out.println("FINISHED!");
    }

    private static void prepareOutputFolder() {
        File dir = new File(PATH_RESULTS);
        if(dir.exists()) dir.delete();
        dir.mkdir();
    }


    private static void runExperiment(String strStream, String strLearner, ExampleStream stream, Regressor learner) throws IOException {

        String fileOutputNameBasic    = prepareFileOutputName(strStream, strLearner, TYPE_BASIC);
        String fileOutputNameWindowed = prepareFileOutputName(strStream, strLearner, TYPE_WINDOWED);

        if(!outputFileExists(fileOutputNameBasic) && !outputFileExists(fileOutputNameWindowed)) {
            // implement a prequential scheme manually to assess two evaluators at once
            WindowRegressionPerformanceEvaluator wEval = new WindowRegressionPerformanceEvaluator();
            BasicRegressionPerformanceEvaluator bEval = new BasicRegressionPerformanceEvaluator();
            wEval.widthOption.setValue(INSTANCES_EVAL);

            // Learning curves
            LearningCurve lcBasic = new LearningCurve("learning evaluation instances");
            LearningCurve lcWindowed = new LearningCurve("learning evaluation instances");

            int instancesSeen = 0;
            boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
            long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            long lastEvaluateStartTime = evaluateStartTime;
            double RAMHours = 0.0;
            while (stream.hasMoreInstances()) {
                Example<Instance> instnc = stream.nextInstance();

                // test
                double pred[] = ((Classifier) learner).getVotesForInstance(instnc);
                wEval.addResult(instnc, pred);
                bEval.addResult(instnc, pred);

                // train
                ((Classifier) learner).trainOnInstance(instnc);

                // updates counter
                instancesSeen++;

                // output percentage
                if (instancesSeen % 1000 == 0) {
                    int pct = (int) (100.0 * instancesSeen / 1000000.0);
                    if (pct % 5 == 0 && pct > 0) {
                        System.out.print(pct + "%\t");
                    }
                }

                // Outputs results if it is the right moment
                if (instancesSeen > 0 && instancesSeen % INSTANCES_EVAL == 0) {
                    long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                    double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                    double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                    double RAMHoursIncrement = ((Classifier) learner).measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                    RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                    RAMHours += RAMHoursIncrement;
                    lastEvaluateStartTime = evaluateTime;
                    lcWindowed.insertEntry(new LearningEvaluation(
                            new Measurement[]{
                                    new Measurement(
                                            "learning evaluation instances",
                                            instancesSeen),
                                    new Measurement(
                                            "evaluation time ("
                                                    + (preciseCPUTiming ? "cpu "
                                                    : "") + "seconds)",
                                            time),
                                    new Measurement(
                                            "model cost (RAM-Hours)",
                                            RAMHours)
                            },
                            wEval, ((Classifier) learner)));
                }
            }
            System.out.println("");

            // Stores the results for the basic evaluator
            lcBasic.insertEntry(new LearningEvaluation(
                    new Measurement[]{
                            new Measurement("learning evaluation instances", instancesSeen),
                    },
                    bEval, ((Classifier) learner)));

            // Output the results for both basic and windowed evaluators
            String contentBasic = lcBasic.toString();
            String contentWindowed = lcWindowed.toString();

            OutputStreamWriter ouw = new OutputStreamWriter(new FileOutputStream(fileOutputNameBasic));
            ouw.write(contentBasic);
            ouw.close();

            ouw = new OutputStreamWriter(new FileOutputStream(fileOutputNameWindowed));
            ouw.write(contentWindowed);
            ouw.close();
        }

    }

    private static String prepareFileOutputName(String strStream, String strLearner, int type) {
        return PATH_RESULTS + (type == TYPE_BASIC ? "BASIC" : "WINDOWED") + "_" +
                strStream.toUpperCase() + "_" + strLearner.toUpperCase() + ".csv";
    }

    private static boolean outputFileExists(String fullPath){
        File f = new File(fullPath);
        if (f.exists())return true;
        return false;
    }

    public static HashMap<String, ArffFileStream> getStreams(){
        // the output streams
        HashMap<String, ArffFileStream> streams = new HashMap<>();

        // retrieves all ARFF files in the given folder
        File dir = new File(PATH_ARFFS);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".arff"));

        for (File f : files){
            ArffFileStream reader = new ArffFileStream();
            reader.arffFileOption.setValue(f.getAbsolutePath());
            reader.prepareForUse();

            streams.put(f.getName().toUpperCase(), reader);
        }

        return streams;
    }

    public static HashMap<String, Regressor> getLearners(){
        HashMap<String, Regressor> learners = new HashMap<>();

        FIMTDD base = new FIMTDD();
        base.prepareForUse();

        HDDM_A_Test warningdetector = new HDDM_A_Test();
        warningdetector.driftConfidenceOption.setValue(10E-4);
        HDDM_A_Test driftdetector = new HDDM_A_Test();
        driftdetector.driftConfidenceOption.setValue(10E-5);
        warningdetector.prepareForUse();
        driftdetector.prepareForUse();
        warningdetector.resetLearning();
        driftdetector.resetLearning();

        int lambdas[] = new int[]{1};
        for(int lambda : lambdas){
            double subspace = 0.1;
            while(subspace <= 0.9) {
                RandomSubspacesRegression rsreg = new RandomSubspacesRegression();
                rsreg.baseLearnerOption.setCurrentObject(base.copy());
                rsreg.lambdaOption.setValue(lambda);
                rsreg.subspacePercentageOption.setValue(subspace);
                rsreg.warningDetectionMethodOption.setCurrentObject(warningdetector);
                rsreg.prepareForUse();
                learners.put("VHPRE-lambda" + lambda + "-subspace" + subspace, rsreg);
                subspace += 0.1;
            }
        }

//        AdaptiveRandomForestRegressor arfreg = new AdaptiveRandomForestRegressor();
//        arfreg.prepareForUse();
//        learners.put("ARFREG", arfreg);
//
//        FIMTDD fimtdd = new FIMTDD();
//        fimtdd.regressionTreeOption.set();
//        fimtdd.prepareForUse();
//        learners.put("FIMTDD", fimtdd);
//
//        ORTO orto = new ORTO();
//        orto.prepareForUse();
//        learners.put("ORTO", orto);
//
//        AMRulesRegressor amrules = new AMRulesRegressor();
//        amrules.prepareForUse();
//        learners.put("AMRules", amrules);
//
//        // SFNR
//        FIMTDD fimtddSFNR = new FIMTDD();
//        fimtddSFNR.regressionTreeOption.set();
//        fimtddSFNR.prepareForUse();
//
//        moa.classifiers.social.SFNRegressor2 sfnr = new SFNRegressor2();
//        sfnr.baseLearnerOption.setCurrentObject(fimtddSFNR);
//        sfnr.prepareForUse();
//        learners.put("SFNR", sfnr);
//
//        // ADDEXP
//        FIMTDD fimtddADDEXP = new FIMTDD();
//        fimtddADDEXP.regressionTreeOption.set();
//        fimtddADDEXP.prepareForUse();
//        AddExpContinuous addexp = new AddExpContinuous();
//        addexp.baseLearnerOption.setCurrentObject(fimtddADDEXP);
//        addexp.prepareForUse();
//        learners.put("ADDEXP", addexp);

        return learners;
    }
}
