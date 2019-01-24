package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.FilteredSparseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.options.ClassOption;

import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RandomSubspacesRegression extends AbstractClassifier implements Regressor {

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'e',
            "Size of the ensemble", 10, 1, 10000);

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'b',
            "Base learner type.", Classifier.class, "trees.FIMTDD");

    // RANDOM SUBSPACES (% of features per learner) 1 = all features to all learners
    public FloatOption subspacePercentageOption = new FloatOption("subspacePercentage", 's',
            "Subspace percentage. 1 = all features are associated with each learner.", 0.7,
            0.0, 1.0);

    // public FlagOption useBaggingOption = new FlagOption("useBagging", 'B',
    //         "Flag to determine whether bagging should be used.");

    public FloatOption lambdaOption = new FloatOption("lambda", 'l',
            "Lambda parameter for bagging.", 1, 1, 10);

    // public MultiChoiceOption baggingTypeOption = new MultiChoiceOption("baggingType",
    //         'T', "", new String[]{"Traditional", "MAE", "RMSE"},
    //         new String[]{"Traditional", "MAE", "RMSE"}, 0);

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
            "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-4");

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
            "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);

//    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'w',
//            "Should use weighted voting?");

    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
            "Should use drift detection? If disabled then bkg learner is also disabled");

    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearner", 'q',
            "Should use bkg learner? If disabled then reset tree immediately.");

    RSRegressionLearner ensemble[];
    long instancesSeen;
    private ExecutorService executor;


    @Override
    public double[] getVotesForInstance(Instance inst) {
        Instance testInstance = inst.copy();
        if(this.ensemble == null) initEnsemble(testInstance);
        double accounted = 0;

        DoubleVector predictions = new DoubleVector();
        // DoubleVector ages = new DoubleVector();
        // DoubleVector performance = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            double currentPrediction = this.ensemble[i].getVotesForInstance(testInstance)[0];
            if(!Double.isNaN(currentPrediction)) {
                // ages.addToValue(i, this.instancesSeen - this.ensemble[i].createdOn);
                // performance.addToValue(i, this.ensemble[i].evaluator.getSquareError());
                predictions.addToValue(i, currentPrediction);
                ++accounted;
            }
        }

        return new double[] {predictions.sumOfValues() / accounted};
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble = null;
        this.instancesSeen = 0;

        // Multi-threading
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1)
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent.
        // this.executor will be null and not used...
        if(numberOfJobs != AdaptiveRandomForest.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);

    }

    protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new RSRegressionLearner[ensembleSize];

        BasicRegressionPerformanceEvaluator regressionEvaluator =
                new BasicRegressionPerformanceEvaluator();

        Classifier baseLearner = (Classifier) getPreparedClassOption(baseLearnerOption);

        for(int i = 0 ; i < ensembleSize ; i++) {
            this.ensemble[i] = new RSRegressionLearner(
                    i,
                    baseLearner.copy(),
                    (BasicRegressionPerformanceEvaluator) regressionEvaluator.copy(),
                    this.instancesSeen,
                    !this.disableBackgroundLearnerOption.isSet(),
                    !this.disableDriftDetectionOption.isSet(),
                    driftDetectionMethodOption,
                    warningDetectionMethodOption,
                    false, subspacePercentageOption.getValue());
        }
    }


    @Override
    public void trainOnInstanceImpl(Instance inst) {

        instancesSeen++;
        if (ensemble == null) initEnsemble(inst);

        for (int i = 0; i < ensemble.length; i++){
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            InstanceExample ex = new InstanceExample(inst);
            ensemble[i].evaluator.addResult(ex, vote.getArrayRef());
            double lambda = lambdaOption.getValue();
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            ensemble[i].trainOnInstance(inst, k, this.instancesSeen);
        }

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public boolean isRandomizable() {
        return true;
    }

    class RSRegressionLearner extends AbstractMOAObject {

        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
        public Classifier learner;
        public boolean isBackgroundLearner;

        // The drift and warning object parameters.
        protected ClassOption driftOption;
        protected ClassOption warningOption;

        // Drift and warning detection
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;

        public boolean useBkgLearner;
        public boolean useDriftDetector;

        // Bkg learner
        protected RSRegressionLearner bkgLearner;
        // Statistics
        public BasicRegressionPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;


        private int featureIndices[] = null;
        private double pctFeatures;

        private void init(int indexOriginal,
                          Classifier instantiatedClassifier,
                          BasicRegressionPerformanceEvaluator evaluatorInstantiated,
                          long instancesSeen,
                          boolean useBkgLearner,
                          boolean useDriftDetector,
                          ClassOption driftOption,
                          ClassOption warningOption,
                          boolean isBackgroundLearner,
                          double pctFeatures) {
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;
            this.pctFeatures = pctFeatures;

            this.learner = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useDriftDetector = useDriftDetector;

            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;

            if (this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }

            // Init Drift Detector for Warning detection.
            if (this.useBkgLearner) {
                this.warningOption = warningOption;
                this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
            }
        }


        public RSRegressionLearner(int indexOriginal,
                                   Classifier instantiatedClassifier,
                                   BasicRegressionPerformanceEvaluator evaluatorInstantiated,
                                   long instancesSeen,
                                   boolean useBkgLearner,
                                   boolean useDriftDetector,
                                   ClassOption driftOption,
                                   ClassOption warningOption,
                                   boolean isBackgroundLearner,
                                   double pctFeatures) {
            init(indexOriginal, instantiatedClassifier,
                    evaluatorInstantiated, instancesSeen,
                    useBkgLearner, useDriftDetector,
                    driftOption, warningOption,
                    isBackgroundLearner,
                    pctFeatures);
        }


        public void reset() {
            if(this.useBkgLearner && this.bkgLearner != null) {
                this.learner = this.bkgLearner.learner;
                this.featureIndices = this.bkgLearner.featureIndices;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;

                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null;
            }
            else {
                this.learner.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset();
        }

        public double[] getVotesForInstance(Instance instnc){
            if(featureIndices == null) featureIndices = getFeatureIndices(pctFeatures,
                    instnc.numAttributes() - 1,
                    instnc.classIndex());
            Instance filteredInstnc = filterInstance(instnc);
            return learner.getVotesForInstance(filteredInstnc);

        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            if(featureIndices == null) featureIndices = getFeatureIndices(pctFeatures,
                    instance.numAttributes() - 1,
                    instance.classIndex());
            Instance filteredInstance = filterInstance(instance);
            for(int i = 0; i < weight; i++) this.learner.trainOnInstance(filteredInstance);


            if(this.bkgLearner != null)
                this.bkgLearner.trainOnInstance(instance, weight, instancesSeen);

            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one.
            if(this.useDriftDetector && !this.isBackgroundLearner) {
//                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                double prediction = this.learner.getVotesForInstance(instance)[0];
                // Check for warning only if useBkgLearner is active
                if(this.useBkgLearner) {
                    // Update the warning detection method
                    this.warningDetectionMethod.input(prediction);
                    // Check if there was a change
                    if(this.warningDetectionMethod.getChange()) {
                        this.lastWarningOn = instancesSeen;
                        this.numberOfWarningsDetected++;
                        // Create a new bkgTree classifier
                        Classifier bkgClassifier = this.learner.copy();
                        bkgClassifier.resetLearning();

                        // Resets the evaluator
                        BasicRegressionPerformanceEvaluator bkgEvaluator = (BasicRegressionPerformanceEvaluator) this.evaluator.copy();
                        bkgEvaluator.reset();

                        // Create a new bkgLearner object
                        this.bkgLearner = new RSRegressionLearner(indexOriginal, bkgClassifier,
                                bkgEvaluator, instancesSeen,
                                this.useBkgLearner, this.useDriftDetector,
                                this.driftOption, this.warningOption, true,
                                subspacePercentageOption.getValue());

                        // Update the warning detection object for the current object
                        // (this effectively resets changes made to the object while it was still a bkg learner).
                        this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
                    }
                }

                /*********** drift detection ***********/

                // Update the DRIFT detection method
                this.driftDetectionMethod.input(prediction);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
                    this.reset();
                }
            }
        }

//        public void trainOnInstance(Instance instnc){
//            Instance filteredInstnc = filterInstance(instnc);
//            this.learner.trainOnInstance(filteredInstnc);
//        }

        public Instance filterInstance(Instance instnc) {
//            Instance filtered = instnc.copy();
//            if(featureIndices != null && featureIndices.length > 0 &&
//                    featureIndices.length < instnc.numAttributes() - 1) {
//                TreeSet<Integer> indices = new TreeSet<>();
//                for(int i = 0; i < featureIndices.length; i++) indices.add(featureIndices[i]);
//                for(int i = 0; i < filtered.numAttributes(); i++){
//                    if(!indices.contains(i) && i != filtered.classIndex()){
//                        filtered.setMissing(i);
//                    }
//                }
//            }
//            return filtered;
            Instance filtered;
            if(featureIndices != null && featureIndices.length > 0 &&
                    featureIndices.length < instnc.numAttributes() - 1){
                int numAttributes = instnc.numAttributes();

                // copies all values including the class
                int indices[] = new int[featureIndices.length + 1];
                double values[] = new double[featureIndices.length + 1];
                for (int i = 0; i < featureIndices.length; i++) {
                    indices[i] = featureIndices[i];
                    values[i] = instnc.value(featureIndices[i]);
                }
                //adds the class index and value
                indices[indices.length - 1] = instnc.classIndex();
                values[indices.length - 1] = instnc.classValue();

                filtered = new FilteredSparseInstance(1.0, values, indices, numAttributes);
                filtered.setDataset(instnc.dataset());
            }else{
                filtered = instnc;
            }
            return filtered;
        }

        private int[] getFeatureIndices(double pctFeatures, int numFeatures, int classIndex) {
            // Selects the features that will be used randomly with equal chance
            // The code also ignores the class index
            int numSelectedFeatures = (int) Math.ceil(pctFeatures * numFeatures);
            TreeSet<Integer> selected = new TreeSet<>();
            while(selected.size() < numSelectedFeatures) {
                int position = classifierRandom.nextInt(numFeatures);
                if(position != classIndex){
                    selected.add(position);
                }
            }

            // builds the final array
            int arr[] = new int[numSelectedFeatures];
            int index = 0;
            for(Integer i : selected){
                arr[index] = i;
                index++;
            }
            // System.out.println(Arrays.toString(arr));
            return arr;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            sb.append("Base learner for Random Subspaces for Regression.");
        }
    }
}
