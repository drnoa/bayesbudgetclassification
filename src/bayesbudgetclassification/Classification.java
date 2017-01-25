package bayesbudgetclassification;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	/*
	 * In the main class we create a bayes classifier and calculate the
	 * probability for all budget positions
	 * 
	 */
	public static void main(String[] args) {

		// get feature vector definition
		Instances featureVector = createFeatureVector();
		// get the trainingsdata
		Instances trainingsSet = buildTrainingsSet();
		// get the data we like to proof
		Instances testData = defineTestdata(featureVector);
		testData.setClassIndex(trainingsSet.numAttributes() - 1);

		try {
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

			/*
			 * now we store all the probabilities in a hashmap togehter with the
			 * relatet budgetpositions
			 */
			HashMap<Double, String> budgetmap = new LinkedHashMap<Double, String>();
			ArrayList<String> budgetPositions = getBudgetPositions();
			for (int i = 0; i < budgetPositions.size(); i++) {
				budgetmap.put(probability[i], budgetPositions.get(i));
			}
			// we try to get the highest possibility of all budget-positions
			double maxValueInMap = Collections.max(budgetmap.keySet());

			int budgetpositionindex = budgetPositions.indexOf(budgetmap.get(maxValueInMap));

			// finaly we add the testet data and the budegetposition back to our trainingsset
			learnNewTrainingData(budgetpositionindex);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/*
	 * In this class we create a
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

	public static void learnNewTrainingData(int learnedclass) throws Exception {
		/*
		 * Load the trainingset add the new trained data to the set and save the
		 * file.
		 */
		Instances testdata = defineTestdata(createFeatureVector());
		Instances trainingsSet = buildTrainingsSet();
		Instance newdataset = testdata.get(0);

		double[] instanceValue1 = newdataset.toDoubleArray();
		double[] instanceValue = Arrays.copyOf(instanceValue1, instanceValue1.length + 1);
		instanceValue[3] = learnedclass;

		try {

			trainingsSet.add(new DenseInstance(1.0, instanceValue));

			BufferedWriter writer = new BufferedWriter(new FileWriter("learned_budget.arff"));
			writer.write(trainingsSet.toString());
			writer.flush();
			writer.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/*
	 * 
	 */
	public static Instances defineTestdata(Instances testData) {

		// set testset
		double[] instanceValue1 = new double[testData.numAttributes() - 1];

		instanceValue1[0] = 1;
		instanceValue1[1] = 0;
		instanceValue1[2] = 1;
		testData.add(new DenseInstance(1.0, instanceValue1));

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

		ArrayList<String> classVal = getBudgetPositions();
		attribute.add(new Attribute("Klasse", classVal));

		Instances featureVectorDefinition = new Instances("TestInstance", attribute, 0);

		return featureVectorDefinition;
	}

	public static ArrayList<String> getBudgetPositions() {

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

		return classVal;
	}

}
