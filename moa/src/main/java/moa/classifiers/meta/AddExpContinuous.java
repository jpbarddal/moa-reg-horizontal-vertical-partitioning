/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.options.ClassOption;

import java.util.ArrayList;
//import weka.core.Instance;

/**
 *
 * @author jeanpaul
 */
public class AddExpContinuous extends AbstractClassifier implements Regressor {

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classificador base para criacao de experts", Regressor.class, "trees.FIMTDD");    
    public FloatOption betaOption = new FloatOption("Beta", 'b',
            "Constante multiplicativa para decrementar o peso dos experts", 0.5f, 0.001f, 1.0f);
    public FloatOption gammaOption = new FloatOption("Gamma", 'g',
            "Constante multiplicativa para o peso de novos experts", 0.5f, 0.001f, 1.0f);
    public FloatOption taoOption = new FloatOption("Tao", 't',
            "Constante de perda requerida para adicionar um novo expert", 0.5f, 0.001f, 1.0f);    
    public IntOption maxExpertsOption = new IntOption("K", 'k', 
            "Constante K para remoção de experts", 1, 1, 1000000);
    
    
    //atributos privados    
    private ArrayList<ExpertAddExp> experts;

    public AddExpContinuous() {
        resetLearningImpl();
    }
        
    @Override
    public void resetLearningImpl() {
        this.experts = new ArrayList<ExpertAddExp>();        
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        for(ExpertAddExp r : experts){
            r.trainOnInstance(instnc);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement measures[] = new Measurement[1];
        Measurement qtdNodes = new Measurement("# experts", experts.size());
        measures[0] = qtdNodes;
        return measures;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
        sb.append("AddExp for Continuous Classes - KOLTER AND MALOOF ICML");
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance instnc) {
        if(experts.isEmpty()){
            ExpertAddExp exp = new ExpertAddExp((Classifier) getPreparedClassOption(baseLearnerOption));
            exp.expert.prepareForUse();
            exp.expert.resetLearning();
            exp.trainOnInstance(instnc);
            experts.add(exp);
        }
        
        //get predictions from experts
        ArrayList<Double> predictions = new ArrayList<Double>();
        for(ExpertAddExp r : experts){
            double votes[] = r.getVotesForInstance(instnc);
            predictions.add(r.getVotesForInstance(instnc)[0]);
        }
        
        
        //calculates the output prediction
        double yObt = 0.0;
        double sumVotes = 0.0;
        double sumWeights = 0.0;
        for(int i = 0; i < experts.size(); i++){
            sumVotes += predictions.get(i) * experts.get(i).getWeight();
            sumWeights += experts.get(i).getWeight();
        }
        yObt = sumVotes / sumWeights;
        
        
        //calculates the suffer loss
        double sufferedLoss = Math.abs(yObt - instnc.classValue());
        
        //updates expert weights
        for(int i = 0; i < experts.size(); i++){
            double w = experts.get(i).getWeight();
            double beta = betaOption.getValue();
            double prediction = predictions.get(i);
            experts.get(i).setWeight(w * Math.pow(beta, (prediction - instnc.classValue())));
        }
        
        //if (suffer loss) > tao -> add a new expert
        if(sufferedLoss > taoOption.getValue()){
            ExpertAddExp newExpert = new ExpertAddExp((Classifier) getPreparedClassOption(baseLearnerOption));
            //calculates the new expert weight
            double newExpWeight = 0.0;
            
            for(Double p : predictions){
                newExpWeight += gammaOption.getValue() * sufferedLoss;
            }
            
            //sets it to the expert
            newExpert.setWeight(newExpWeight);
            
            //adds it to the ensemble
            experts.add(newExpert);
        }
        
        
        //pruning
        if(experts.size() > maxExpertsOption.getValue()){
            //choose the expert to be removed
            ExpertAddExp toRemove = experts.get(0);
            for(ExpertAddExp ite : experts){
                if(ite.getWeight() < toRemove.getWeight()){
                    toRemove = ite;
                }
            }
            
            //remove it from the ensemble
            experts.remove(toRemove);            
        }
        
        double votes[] = new double[1];
        votes[0] = yObt;
        return votes;
    }


    public class ExpertAddExp {

        private double weight;
        private Classifier expert;

        public ExpertAddExp(Classifier expert) {
            this.expert = expert;
            this.weight = 1.0;
        }

        public double getWeight() {
            return weight;
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }

        public Classifier getExpert() {
            return expert;
        }

        public void setExpert(Classifier expert) {
            this.expert = expert;
        }

        public void trainOnInstance(Instance instnc) {
            expert.trainOnInstance(instnc);
        }

        public double[] getVotesForInstance(Instance instnc) {
            return expert.getVotesForInstance(instnc);
        }

    }


}
