package bayesbudgetclassification;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.SimulatedAnnealing;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	public static void main(String[] args) {
		try {
		    DataSource source = new DataSource("learned_budget.arff");
		    Instances instances = source.getDataSet();
		    instances.setClassIndex(0);

		    BayesNet bayesnet = new BayesNet();
		    
		    SearchAlgorithm searchAlgorithm=new SimulatedAnnealing();
		    bayesnet.setSearchAlgorithm(searchAlgorithm);
		    bayesnet.buildClassifier(instances);
		    Evaluation evaluation = new Evaluation(instances);
		    evaluation.evaluateModel(bayesnet, instances);
		    System.out.println(evaluation.toSummaryString("Resultate\n",false));
		    System.out.println(bayesnet.getNodeValue(1, 0));
		    System.out.println(bayesnet.getProbability(3, 2, 0));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
