package bayesbudgetclassification;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	public static void main(String[] args) {
		try {
			/*
			 * Load the trainingset. In the example this are all the already
			 * classified payments.
			 */
		    DataSource source = new DataSource("learned_budget.arff");
		    Instances data = source.getDataSet();
		    data.setClassIndex(data.numAttributes() - 1);

		    /*
		     * Load the payment we like to classify
		     */
		    DataSource totest = new DataSource("to_test.arff");
		    Instances testdata = totest.getDataSet();
		    testdata.setClassIndex(data.numAttributes() - 1);
		    
		    /*
		     * Create the classifier from weka
		     */
		    NaiveBayes naivebayes = new NaiveBayes();
		    naivebayes.buildClassifier(data);
		    
		    /*
		     * Evaluate the possibilities
		     * The result shows that only one record of the testdata is correct.
		     * @Todo: Implement a loop to check each record
		     */		    
		    Evaluation eval_train = new Evaluation(testdata);
		    eval_train.evaluateModel(naivebayes,testdata);
		    
		    System.out.println(eval_train.toSummaryString("Resultate\n",false));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
