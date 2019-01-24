package moa.classifiers.meta;

import as.graph.Graph;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import moa.options.ClassOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;
//import weka.core.Instance;

/**
 *
 * @author Jean Paul Barddal
 */
public class SFNRegressor extends AbstractClassifier implements Regressor {

    //Definição de nó para o grafo
    public class SFNRVertex {

        //base learner
        private Classifier classifier;

        //Classifier ID on the network
        private int id;

        //Classifier name on the network
        private String nome;

        //declaracao de instancias totais
        private int instancesSeen;

        private int instancesPeriod;

        private String errorMethod;

        private double sumError;
        
        

        //construtor
        public SFNRVertex() {
        }

        public SFNRVertex(int id, Classifier classifier, String errorMethod) {
            this.id = id;
            this.nome = "Node " + id;
            this.classifier = classifier.copy();
            this.classifier.resetLearning();
            this.errorMethod = errorMethod;
            this.sumError = 0.0;

        }

        //funções
        public double[] getVotesForInstance(Instance instance) {
            double vote[] = classifier.getVotesForInstance(instance);
            double error = 0.0;
            if (errorMethod.equals("MAE")) {
                error = Math.abs(instance.classValue() - vote[0]);
            } else if (errorMethod.equals("MSE")) {
                error = Math.pow(Math.abs(instance.classValue() - vote[0]), 2);
            }
            sumError += error;
            instancesSeen++;
            instancesPeriod++;
//            System.out.println(vote[0]);
            return vote;
        }

        public void trainOnInstance(Instance instance) {
            classifier.trainOnInstance(instance);
        }

        public double getErrorRate() {
            double error;
            error = sumError / instancesPeriod;
            return error;
        }

        public void cleanStats() {
            sumError = 0.0;
            instancesPeriod = 0;
        }

        public boolean correctlyClassifies(Instance instnc) {

            return classifier.correctlyClassifies(instnc);
        }

    }

    //////////////////////////////////////
    // VARIÁVEIS AUXILIARES PARA INTERFACE
    //////////////////////////////////////
    private final String errorMethod[] = {"MAE", "MSE"}; //MAE stands for Mean absolute error and MSE stands for Mean squared error
    private final String removalMethod[] = {"Elitism", "AVG", "STDDEV+AVG"};
    private final String optDriftDetectionMethod[] = {"MaxErrorThreshold", "ADWIN"};
    private final String metrics[] = {"Betweenness", "Closeness", "Degree", "Pagerank", "Eigenvector"};
    //////////////////////////////////////
    // VARIÁVEIS DE OPTIONS
    //////////////////////////////////////    
    public MultiChoiceOption errorMethodOption = new MultiChoiceOption("errorMethodOption", 'q', "Determine which error will be used.", errorMethod, errorMethod, 0);
    public MultiChoiceOption removalMethodOption = new MultiChoiceOption("removalMethodOption", 'o', "Determine which method for experts removal will be used.", removalMethod, removalMethod, 0);
    public MultiChoiceOption driftDetectionMethodOption = new MultiChoiceOption("driftDetectionMethod", 'd', "Determine which drift detection method will be used.", optDriftDetectionMethod, optDriftDetectionMethod, 0);
    public FloatOption maxErrorThresholdOption = new FloatOption("maxErrorThreshold", 'e', "Determine the maximum error allowed to maintain the concept learned.", 0.08f, 0.0f, Float.MAX_VALUE);
    public ClassOption baseLearnerOption = new ClassOption("baseLeaner", 'l', "Classifier to train.", Classifier.class, "trees.FIMTDD");
    public MultiChoiceOption adoptedMetricOption = new MultiChoiceOption("adoptedMetric", 'm', "Determine which metric will be used for voting on instances.", metrics, metrics, 0);
    public IntOption updatePeriodOption = new IntOption("updatePeriodOption", 'u', "Define how many periods the network remains without any udpate.", 1000, 0, 50000);
    public IntOption kMaxOption
            = new IntOption("kMax", 'k',
                    "Determines the maximum amount of nodes in the network.",
                    3, 3, 1000);
    /////////////////////////////////////
    // VARIÁVEIS INTERNAS
    /////////////////////////////////////
    private transient Graph<SFNRVertex, Integer> network
            = new Graph<SFNRVertex, Integer>(adoptedMetricOption.getValueAsCLIString());

    private int lastID = 0;
    private int instancesSeen = 0;
    private ArrayList<Instance> badInstances
            = new ArrayList<Instance>();
    private int instancesInThisPeriod = 0;
    private double sumError = 0.0;
    private HashMap<Integer, ADWIN> adError = new HashMap<Integer, ADWIN>();
    private HashMap<Instance, Double> relationRealObtained;
    
    //
    @Override
    public void resetLearningImpl() {
        lastID = 0;
        network = new Graph<SFNRVertex, Integer>(adoptedMetricOption.getValueAsCLIString());
        instancesSeen = 0;
        badInstances = new ArrayList<Instance>();
        sumError = 0.0;
        adError = new HashMap<Integer, ADWIN>();
        this.relationRealObtained = new HashMap<>();
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        for (Integer iterator : network.getNodesIDs()) {
            ((SFNRVertex) network.getNode(iterator)).trainOnInstance(instnc);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement measures[] = new Measurement[2];
        Measurement qtdNodes = new Measurement("# nodes", network.getNodesQuantity());
        Measurement rmsle = new Measurement("RMSLE", calculateRMSLE());
        measures[0] = qtdNodes;
        measures[1] = rmsle;
        return measures;
    }
    
    private double calculateRMSLE(){
        double rmsle = 0.0;
        
        for(Instance ite : this.relationRealObtained.keySet()){
            double inc = Math.pow(Math.log((double) this.relationRealObtained.get(ite) + 1) - Math.log((double) ite.classValue() + 1), 2);
            //System.out.println("real: " + ite.classValue() + " obt: " + this.relationRealObtained.get(ite) + " inc: " + inc);
            if(Double.isNaN(inc)){
                rmsle += 0.0; 
            }else{
                rmsle += Math.pow(Math.log((double) this.relationRealObtained.get(ite) + 1) - Math.log((double) ite.classValue() + 1), 2);
            }
        }
        
        rmsle = rmsle / (double)this.relationRealObtained.size();
        rmsle = Math.sqrt(rmsle);
        
        //reset variables
        this.relationRealObtained.clear();
        
        //return
        return rmsle;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance instnc) {
        //caso não existam classificadores, adiciona o primeiro
        if (this.network.getNodesQuantity() == 0) {
            ArrayList<Instance> arr = new ArrayList<Instance>();
            arr.add(instnc);
            SFNRVertex newVertice = this.instantiateNewVertex(arr);
            this.addVertex(newVertice);
        }

        double votes[] = new double[1];
        votes[0] = 0.0;
        double sum = 0.0;
        for (Integer v : this.network.getNodesIDs()) {
            double metrica = network.getCentralityMetric(v);
            votes[0] += metrica * network.getNode(v).getVotesForInstance(instnc)[0];
            sum += metrica;
        }
        votes[0] /= sum;
        if(instnc.classIsMissing()){
//            System.out.println(votes[0]);
        }
        this.relationRealObtained.put(instnc, votes[0]);
        
        double error = 0.0;
        if (this.errorMethodOption.getValueAsCLIString().equals("MAE")) {
            error = Math.abs(votes[0] - instnc.classValue());
        } else if (this.errorMethodOption.getValueAsCLIString().equals("MSE")) {
            Math.pow(Math.abs(votes[0] - instnc.classValue()), 2);
        }
        this.sumError = this.sumError + error;

        
        if (this.instancesInThisPeriod > this.updatePeriodOption.getValue()) {
            this.badInstances.add(instnc);
        }

        //checks whether a drift occured
        boolean driftDetected = false;
        if (instancesSeen % updatePeriodOption.getValue() == 0 && driftDetectionMethodOption.getValueAsCLIString().equals("MaxErrorThreshold")) {
            driftDetected = (sumError / this.instancesInThisPeriod > maxErrorThresholdOption.getValue());
        } else if (driftDetectionMethodOption.getValueAsCLIString().equals("ADWIN")) {
            ArrayList<Integer> vertices = new ArrayList<Integer>(network.getNodesIDs());
            for (int i = 0; i < network.getNodesQuantity(); i++) {
                double errEstim = adError.get(vertices.get(i)).getEstimation();
                boolean correctlyClassifies = this.network.getNode(vertices.get(i)).correctlyClassifies(instnc);
                if (adError.get(vertices.get(i)).setInput(correctlyClassifies ? 0 : 1)) {
                    if (adError.get(vertices.get(i)).getEstimation() > errEstim) {
                        driftDetected = true;
                    }
                }
            }

        }

        if (driftDetected) {
            // In case ADWIN algorithm has found a drift
            // or the period is over            
            updateNetwork();

            //cleans overall stats
            cleanStats();
            //cleans every node stats
            for (Integer v : this.network.getNodesIDs()) {
                network.getNode(v).cleanStats();
            }
        } else if (!driftDetected && this.instancesSeen % this.updatePeriodOption.getValue() == 0) {
            cleanStats();
        }

        this.instancesInThisPeriod++;
        this.instancesSeen++;

        return votes;
    }

    @Override
    public String getPurposeString() {
        //return "";
        return "SFNRegressor: A Scale-free Network Algorithm for the Regression Task.";
    }

    ///////////////////////////////////
    // MUTATORS 
    ///////////////////////////////////
    private void cleanStats() {
        sumError = 0.0;
        this.instancesInThisPeriod = 0;
        badInstances.clear();
    }

    private void updateNetwork() {
        //choose vertices for removal
        ArrayList<Integer> toRemove = this.chooseVerticesForRemoval();

        //remove them
        this.removeVertices(toRemove);
        //adds the new expert to the network
        if (sumError / this.instancesInThisPeriod > this.maxErrorThresholdOption.getValue()) {
            SFNRVertex newClassifier = instantiateNewVertex(this.badInstances);
            this.addVertex(newClassifier);
        }

    }

    private void addVertex(SFNRVertex newVertex) {
        //insere o novo classificador na rede
        if (newVertex != null) {
            // we determine which node will establish 
            // a connection to this new node
            int neighbor = chooseNeighbor();

            //we add the newVertex to the network
            //and attach it to the choosen neighbor
            network.addNode(lastID++, newVertex);
            network.setEdge(neighbor, lastID - 1, 1);
            if (this.driftDetectionMethodOption.getValueAsCLIString().equals("ADWIN")) {
                this.adError.put(lastID - 1, new ADWIN());
            }
        }
    }

    private int chooseNeighbor() {
        int choosen = -1;

        if (network.getNodesQuantity() == 1) {
            return network.getNodesIDs().get(0);
        }

        double errors = 0.0;
        for (Integer ite : network.getNodesIDs()) {
            errors += network.getNode(ite).getErrorRate();
        }

        double threshold = (Math.random() * errors);
        double sum = 0.0;
        ArrayList<Integer> all = new ArrayList<Integer>(network.getNodesIDs());
        Collections.shuffle(all);
        for (Integer v : all) {
            sum += (errors - network.getNode(v).getErrorRate());
            if (sum >= threshold) {
                choosen = v;
                break;
            }
        }

        return choosen;
    }

    private void removeVertices(ArrayList<Integer> vIDs) {
        int id = -1;
        if (!vIDs.isEmpty()) {
            id = vIDs.get(0);
        }
        while (id != -1) {

            ArrayList<Integer> neighbors
                    = new ArrayList<Integer>(network.getNeighborsIDs(id));
            network.removeNode(id);
            //remove o detector relativo do adwin
            adError.remove(id);

            Collections.sort(neighbors,
                    new Comparator() {
                        public int compare(Object o1, Object o2) {
                            SFNRVertex p1 = network.getNode((Integer) o1);
                            SFNRVertex p2 = network.getNode((Integer) o2);
                            return p1.getErrorRate() < p2.getErrorRate() ? +1 : (p1.getErrorRate() > p2.getErrorRate() ? -1 : 0);
                        }
                    });

            int mainVertex = neighbors.get(0);

            for (int ite : neighbors) {
                if (ite != mainVertex) {
                    network.setEdge(mainVertex, ite, 1);
                }
            }

            vIDs.remove(0);

            if (!vIDs.isEmpty()) {
                id = vIDs.get(0);
            } else {
                id = -1;
            }
        }
    }

    private ArrayList<Integer> chooseVerticesForRemoval() {
        ArrayList<Integer> ret = new ArrayList<Integer>();
        if (network.getNodesQuantity() > kMaxOption.getValue()) {
            if (removalMethodOption.getValueAsCLIString().equals("Elitism")) {
                ArrayList<Integer> aux = new ArrayList<Integer>(network.getNodesIDs());

                //ordena os vertices a serem removidos em ordem CRESCENTE de hit rate
                Collections.sort(aux,
                        new Comparator() {
                            public int compare(Object o1, Object o2) {
                                SFNRVertex p1 = network.getNode((Integer) o1);
                                SFNRVertex p2 = network.getNode((Integer) o2);
                                return (p1.getErrorRate() > p2.getErrorRate() ? -1 : (p1.getErrorRate() < p2.getErrorRate() ? +1 : 0));
                            }
                        });
                ret.add(aux.get(0));

            } else if (removalMethodOption.getValueAsCLIString().equals("AVG")) {

                double avg = 0.0;
                for (Integer v : network.getNodesIDs()) {
                    avg += network.getNode(v).getErrorRate();
                }
                //if v.geterrorrate > avg
                for (Integer v : network.getNodesIDs()) {
                    if (network.getNode(v).getErrorRate() > avg) {
                        ret.add(v);
                    }

                }
            } else if (removalMethodOption.getValueAsCLIString().equals("STDDEV+AVG")) {

                double avg = 0.0;
                for (Integer v : network.getNodesIDs()) {
                    avg += network.getNode(v).getErrorRate();
                }
                double stddev = 0.0;
                for (Integer v : network.getNodesIDs()) {
                    stddev += Math.pow(network.getNode(v).getErrorRate() - avg, 2);
                }
                stddev = Math.sqrt(stddev);
                //if v.geterrorrate > avg + stddev
                for (Integer v : network.getNodesIDs()) {
                    if (network.getNode(v).getErrorRate() > avg + stddev) {
                        ret.add(v);
                    }
                }

            }
        }
        return ret;
    }

    private SFNRVertex instantiateNewVertex(ArrayList<Instance> arr) {
        SFNRVertex newVertex = new SFNRVertex(lastID++, (Classifier) getPreparedClassOption(this.baseLearnerOption), this.errorMethodOption.getChosenLabel());
        for (Instance instance : arr) {
            newVertex.trainOnInstance(instance);
        }
        return newVertex;
    }
}
