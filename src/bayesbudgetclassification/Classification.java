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
	 */
	public static void main(String[] args) {

		// get feature vector definition
		Instances featureVector = createFeatureVector();
		// get the trainingsdata
		Instances trainingsSet = buildTrainingsSet();

		System.out.println("*The Trainingsset we use looks like this:*");
		System.out.println(trainingsSet);

		// get the data we like to proof
		Instances testData = defineTestdata(featureVector);
		testData.setClassIndex(trainingsSet.numAttributes() - 1);

		System.out.println("\n*The payment i like to check looks like this:*");
		System.out.println(trainingsSet);

		try {
			/*
			 * Create the classifier
			 */
			NaiveBayes naivebayes = new NaiveBayes();
			/*
			 * give the trainingdataset to the classifier
			 */
			naivebayes.buildClassifier(trainingsSet);
			/*
			 * get the probability for each class this gives us the probability
			 * for each budgetposition
			 */
			double[] probability = naivebayes.distributionForInstance(testData.firstInstance());

			System.out.println("\n*the bayes classification calculates the following results*");

			/*
			 * now we store all the probabilities in a hashmap togehter with the
			 * relatet budgetpositions
			 */
			HashMap<Double, String> budgetmap = new LinkedHashMap<Double, String>();
			ArrayList<String> budgetPositions = getBudgetPositions();
			for (int i = 0; i < budgetPositions.size(); i++) {
				budgetmap.put(probability[i], budgetPositions.get(i));
				System.out.println("The probability that my testdata belongs to the budgetposition "
						+ budgetPositions.get(i) + " is " + Math.round(probability[i] * 10000) / 10000.0 * 100 + "%");
			}
			// we try to get the highest possibility of all budget-positions
			double maxValueInMap = Collections.max(budgetmap.keySet());

			System.out.println("Based on the calculation we select the highest probability whitch is "
					+ Math.round(maxValueInMap * 10000) / 10000.0 * 100 + "%");

			int budgetpositionindex = budgetPositions.indexOf(budgetmap.get(maxValueInMap));

			// finaly we add the testet data and the budegetposition back to our
			// trainingsset
			learnNewTrainingData(budgetpositionindex);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/*
	 * In this class we create a
	 */
	private static Instances buildTrainingsSet() {

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
	private static void learnNewTrainingData(int learnedclass) throws Exception {
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

			System.out.println("\nThe dataset we get looks like this: "+trainingsSet.lastInstance());

			BufferedWriter writer = new BufferedWriter(new FileWriter("learned_budget.arff"));
			writer.write(trainingsSet.toString());
			writer.flush();
			writer.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/*
	 * defines the Testdataset. This is the payment we like to check an classify
	 */
	private static Instances defineTestdata(Instances testData) {

		// set testset
		double[] instanceValue1 = new double[testData.numAttributes() - 1];

		instanceValue1[0] = 1; // Dienstag
		instanceValue1[1] = 0; // Lebensmittel
		instanceValue1[2] = 1; // klein
		testData.add(new DenseInstance(1.0, instanceValue1));

		return testData;
	}

	/*
	 * This method generates the definition of our vectors and sets all possible
	 * attributes and values of them
	 * 
	 */
	private static Instances createFeatureVector() {

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

	/*
	 * defines the possible budet positions
	 */
	private static ArrayList<String> getBudgetPositions() {

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
