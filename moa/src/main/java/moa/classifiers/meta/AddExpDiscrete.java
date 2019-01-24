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
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;
import java.util.ArrayList;

/**
 *
 * @author jeanpaul
 */
public class AddExpDiscrete extends AbstractClassifier  {

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classificador base para criacao de experts", Classifier.class, "trees.HoeffdingTree");    
    public FloatOption betaOption = new FloatOption("Beta", 'b',
            "Constante multiplicativa para decrementar o peso dos experts", 0.5f, 0.001f, 1.0f);
    public FloatOption gammaOption = new FloatOption("Gamma", 'g',
            "Constante multiplicativa para o peso de novos experts", 0.5f, 0.001f, 1.0f);       
    public IntOption maxExpertsOption = new IntOption("maxExperts", 'k', 
            "Constante K para remoção de experts", 1, 1, 1000000);
    
    
    //atributos privados    
    private ArrayList<ExpertAddExp> experts;

    public AddExpDiscrete() {
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
        //output
        double votes[] = new double[instnc.numClasses()];
        
        if(experts.isEmpty()){
            ExpertAddExp exp = new ExpertAddExp((Classifier) getPreparedClassOption(baseLearnerOption));
            exp.trainOnInstance(instnc);
            experts.add(exp);
        }
        
        //get predictions from experts
        ArrayList<Integer> previsoes = new ArrayList<Integer>();        
        //calculates the output prediction
        for(ExpertAddExp e : experts){
            previsoes.add(Utils.maxIndex(e.getVotesForInstance(instnc)));
            votes[Utils.maxIndex(e.getVotesForInstance(instnc))] += e.getWeight();
        }
        
        //updates expert weights
        for(int i = 0; i < experts.size(); i++){
            if(Utils.maxIndex(votes) != previsoes.get(i)){
                //atualiza o peso antigo
                experts.get(i).setWeight(betaOption.getValue() * experts.get(i).getWeight() );
            }
            
        }
                
        //adiciona novo expert caso a previsão seja errada
        double valorEsperado = instnc.classValue();
        int indiceObtido = Utils.maxIndex(votes);
        if(valorEsperado != indiceObtido){
            ExpertAddExp exp = new ExpertAddExp((Classifier) getPreparedClassOption(baseLearnerOption));
            double sumWeights = 0.0;
            for(ExpertAddExp e : experts){
                sumWeights += e.getWeight();
            }
            exp.setWeight(gammaOption.getValue() * sumWeights);            
            experts.add(exp);
            System.out.println("ERREI");
        }
        
        //pruning
        /*if(experts.size() > maxExpertsOption.getValue()){
            //choose the expert to be removed
            ExpertAddExp toRemove = experts.get(0);
            for(ExpertAddExp ite : experts){
                if(ite.getWeight() < toRemove.getWeight()){
                    toRemove = ite;
                }
            }
            
            //remove it from the ensemble
            experts.remove(toRemove);            
        }*/
                
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
