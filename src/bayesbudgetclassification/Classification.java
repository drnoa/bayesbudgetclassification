package bayesbudgetclassification;

import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	public static void main(String[] args) {

		// get feature vector definition
		Instances featureVector = createFeatureVector();
		Instances testData = defineTestdata(featureVector);
		Instances trainingsSet = buildTrainingsSet();

		try {

			testData.setClassIndex(trainingsSet.numAttributes() - 1);
			/*
			 * Create the classifier from weka
			 */
			NaiveBayes naivebayes = new NaiveBayes();
			/*
			 * give the classifier the trainingdata
			 */
			naivebayes.buildClassifier(trainingsSet);
			/*
			 * get the probability for each class
			 */
			double[] probability = naivebayes.distributionForInstance(testData.firstInstance());
			
			
			System.out.println(Arrays.toString(probability));

			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/*
	 * 
	 */
	public static Instances buildTrainingsSet() {

		/*
		 * Load the trainingset. In the example this are all the already
		 * classified payments.
		 */
		DataSource source;
		try {
			source = new DataSource("learned_budget.arff");
			Instances data = source.getDataSet();
			data.setClassIndex(data.numAttributes() - 1);
			return data;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/*
	 * 
	 */
	public static Instances defineTestdata(Instances testData) {

		// set testset
		double[] instanceValue1 = new double[testData.numAttributes()-1];

		instanceValue1[0] = 1;
		instanceValue1[1] = 0;
		instanceValue1[2] = 1;
		testData.add(new DenseInstance(1.0, instanceValue1));

		System.out.println(testData);

		return testData;
	}

	/*
	 * 
	 */
	public static Instances createFeatureVector() {

		ArrayList<Attribute> attribute = new ArrayList<Attribute>(5);

		// Declare Attributes
		ArrayList<String> weekday = new ArrayList<String>();
		weekday.add("Montag");
		weekday.add("Dienstag");
		weekday.add("Mittwoch");
		weekday.add("Donnerstag");
		weekday.add("Freitag");
		weekday.add("Samstag");
		weekday.add("Sonntag");
		attribute.add(new Attribute("Wochentag", weekday));

		ArrayList<String> store = new ArrayList<String>();
		store.add("Lebensmittel");
		store.add("Moebel");
		store.add("Gesundheit");
		store.add("Verkehr");
		store.add("Mobilfunk");
		store.add("Radio");
		store.add("Energie");
		store.add("Versicherungen");
		store.add("Steuern");
		store.add("Bankomat");
		store.add("Andere");
		attribute.add(new Attribute("Geschäft", store));

		ArrayList<String> amount = new ArrayList<String>();
		amount.add("mikro");
		amount.add("klein");
		amount.add("mittel");
		amount.add("gross");
		amount.add("riesig");
		attribute.add(new Attribute("Betrag", amount));

		ArrayList<String> classVal = new ArrayList<String>();
		classVal.add("Nahrungsmittel");
		classVal.add("Persönliche Ausgaben");
		classVal.add("Hobby");
		classVal.add("Freizeit");
		classVal.add("Gesundheitskosten");
		classVal.add("Bekleidung");
		classVal.add("Arzt");
		classVal.add("Reserve");
		classVal.add("Miete");
		classVal.add("Steuern");
		classVal.add("Krankenkasse");
		attribute.add(new Attribute("Klasse", classVal));

		Instances featureVectorDefinition = new Instances("TestInstance", attribute, 0);

		return featureVectorDefinition;
	}

}
