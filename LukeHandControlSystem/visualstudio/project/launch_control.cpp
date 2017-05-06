//launch_control.cpp: Used to start up the control system. Listens to the 
//Myo, filters and classifies the data.

//Myo API includes
#define _USE_MATHDEFINES
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <myo/myo.hpp>
#include <vector>
#include <stdlib.h>
using namespace std;

#include <cmath>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <Windows.h>


//Include OPENCV Machine Learning Libraries
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\ml\ml.hpp>
#include <armadillo>
using namespace cv;

using namespace arma;

//Define States.
#define INIT 1
#define LOG 2
#define FILTER 3
#define EXTRACT_FEATURES 4
#define TRAIN 5
#define CROSS_VALIDATE 6
#define CLASSIFY 7
#define SEND2SOCKET 8
#define EXIT 9
#define DEBUG 10

//Define Classes
#define REST 1;
#define FIST 2;
#define WAVE_IN 3;
#define WAVE_OUT 4;
#define THUMBS_UP 5;
#define OK 6;
#define OPEN_HAND 7;
#define PEACE 8;
#define ROCK_ON 9;
//Will add more gesture definitions here.


//Set Parameters for the training data
int num_classes = 5;
int FS = 200;
int WIN_SIZE = 500;

//typedef std::vector<int> stdvec;

class DataCollector : public myo::DeviceListener {
public:
    DataCollector()
    : emgSamples()
    {
		//openFiles();
    }

	void openFiles(const string& name) {
		time_t timestamp = std::time(0);

		// Open file for EMG log
		if (emgFile.is_open()) {
			emgFile.close();
		}
		std::ostringstream emgFileString;
		emgFileString << name << timestamp << ".csv";
		emgFile.open(emgFileString.str(), std::ios::out);
		//emgFile << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

		
	}

	void closeFiles() {
		emgFile.close();
	}

    // If Myo is disconnected, start 
    void onUnpair(myo::Myo* myo, uint64_t timestamp)
    {
        emgSamples.fill(0);
    }
    // onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
    void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
    {
        for (int i = 0; i < 8; i++) {
            emgSamples[i] = emg[i];
			emgFile << ',' << static_cast<int>(emg[i]);
        }
		emgFile << std::endl;
    }



    // There are other virtual functions in DeviceListener that we could override here, like onAccelerometerData().
    // For this example, the functions overridden above are sufficient.

    // We define this function to print the current values that were updated by the on...() functions above.
    void print()
    {
        // Clear the current line
        std::cout << '\n';
		
        // Print out the EMG data.
        for (size_t i = 0; i < emgSamples.size(); i++) {
            std::ostringstream oss;
            oss << static_cast<int>(emgSamples[i]);
            std::string emgString = oss.str();

            std::cout << '[' << emgString << std::string(4 - emgString.size(), ' ') << ']';
			
        }

        std::cout << std::flush;
    }

    // The values of this array is set by onEmgData() above.
    std::array<int8_t, 8> emgSamples;
	std::ofstream emgFile;

};

//Helper functions
typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;

stdvecvec mat2StdVec(const arma::mat &A) {
	stdvecvec V(A.n_rows);
	for (size_t i = 0; i < A.n_rows; ++i) {
		V[i] = arma::conv_to< stdvec >::from(A.row(i));
	};
	return V;
}

double** vec2Array2D(const vector<vector<double> > &vals, double N, double M){
	double** temp;
	temp = new double*[N];
	for (unsigned i = 0; (i < N); i++)
	{
		temp[i] = new double[M];
		for (unsigned j = 0; (j < M); j++)
		{
			temp[i][j] = vals[i][j];
		}
	}
	return temp;
}

double* vec2Array(stdvec& v) {
	//double *array = &v[0];
	return &v[0];
}


vector<int> createVector(int startval, int stepval, int endval) {
	vector<int> result;
	while (startval <= endval) {
		result.push_back(startval);
		startval += stepval;
	}
	return result;
}
int calculateNumWindows(const vector<int>& window, int sampling_freq, int win_length, int win_disp) {
	int num_windows = 0;
	int start = 1 + win_length*sampling_freq;
	int step = win_disp*sampling_freq;
	int end = window.size() + win_length*sampling_freq;
	vector<int> window_vec = createVector(start, step, end);
	for (int j = 0; j < window_vec.size(); ++j) {
		if (window_vec[j] <= window.size() + 1) {
			++num_windows;
		}
	}
	return num_windows;
}

void printVector(const vector<int>& vec) {
	for (int i = 0; i < vec.size(); ++i) {
		cout << vec[i] << endl;
	}
}

int numWindows(const mat& matrix, int fs, int win_size, int win_disp) {
	int mat_size = matrix.n_rows;
	int start = 1 + win_size*fs/1000;
	int step = win_disp*fs/1000;
	int end = mat_size + win_size*fs/1000;
	vector<int> svec = createVector(start, step, end);
	colvec armvec = conv_to< colvec >::from(svec);
	uvec idx = find(armvec <= mat_size + 1);
	return idx.n_rows;
	
}

//Calculate Line length feature of a matrix given a raw data matrixe
mat lineLength(const mat& X) {
	return sum(abs(diff(X)));

}

//Calculate Area Feature Matrix given raw data
mat area(const mat& X) {
	return sum(abs(X));
}

//Calculate Log variance matrix

mat logVar(const mat& X) {
	//cout << "Calculate logvar" << endl;
	//Sleep(2000);
	mat log_variance = log10(var(X, 0, 1));
	//cout << log_variance(0, 0) << endl;
	return sum(log_variance);
}

//Claculates Feature Matrix for a given raw data matrix
mat calculateFeatures(const mat& raw_data, int num_windows, int win_size, int win_disp, int fs) {
	mat ll_mat;
	mat area_mat;
	mat log_var;
	ll_mat.zeros(num_windows,8);
	area_mat.zeros(num_windows, 8);
	log_var.zeros(num_windows, 1);


	for (int i = 1; i <= num_windows; ++i) {
		int p = 1 + win_disp*fs*(i - 1) / 1000;
		int q = 1 + win_disp*fs*(i - 1)/1000 + win_size*fs/1000 - 1;
		//cout << p << endl << q << endl;
		//Sleep(2000);
		mat clip = raw_data.rows(p-1,q-1);
		ll_mat.row(i-1) = lineLength(clip);
		area_mat.row(i-1) = area(clip);
		log_var.row(i - 1) = logVar(clip);
	}
	//add elements horizontally:
	colvec ll_vec = sum(ll_mat, 1);
	colvec area_vec = sum(area_mat, 1);
	colvec logvar_vec = sum(log_var, 1);


	mat feature_mat;
	feature_mat = join_horiz(ll_vec , join_horiz(area_vec,log_var));
	return feature_mat;
}

int main(int argc, char** argv)
{
    // We catch any exceptions that might occur below -- see the catch statement for more details.
    try {

    // First, we create a Hub with our application identifier. Be sure not to use the com.example namespace when
    // publishing your application. The Hub provides access to one or more Myos.
    myo::Hub hub("com.example.Control");

    std::cout << "Attempting to find a Myo..." << std::endl;

    myo::Myo* myo = hub.waitForMyo(10000);

    // If waitForMyo() returned a null pointer, we failed to find a Myo, so exit with an error message.
    if (!myo) {
        throw std::runtime_error("Unable to find a Myo!");
    }

    // We've found a Myo.
    std::cout << "Connected to a Myo armband!" << std::endl << std::endl;

    // Next we enable EMG streaming on the found Myo.
    myo->setStreamEmg(myo::Myo::streamEmgEnabled);
    // Next we construct an instance of our DeviceListener, so that we can register it with the Hub.
    DataCollector collector;

    // Hub::addListener() takes the address of any object whose class inherits from DeviceListener, and will cause
    // Hub::run() to send events to all registered device listeners.
    hub.addListener(&collector);
	// Finally we enter our main loop.
	vector<string> gestures = {"rest","fist","wavein","waveout","thumbsup"};

	////////////////////////////// Moving Feature Extraction Block Here Temporarily/////////////////////
	mat rest, fist, wavein, waveout, thumbsup;
	mat rest_feats, fist_feats, wavein_feats, waveout_feats, thumbsup_feats;
	cout << "Loading Files ..." << endl;
	rest.load("rest.csv", csv_ascii);
	fist.load("fist.csv", csv_ascii);
	wavein.load("wavein.csv", csv_ascii);
	waveout.load("waveout.csv", csv_ascii);
	thumbsup.load("thumbsup.csv", csv_ascii);
	cout << "Cleaning up files ..." << endl;
	rest = rest.cols(1, 8);
	fist = fist.cols(1, 8);
	wavein = wavein.cols(1, 8);
	waveout = waveout.cols(1, 8);
	thumbsup = thumbsup.cols(1, 8);
	cout << "Calculating features ..." << endl;
	int nrest_wins, nfist_wins, nwaveout_wins, nwavein_wins, nthumbsup_wins;
	//////////////////    Train Rest ///////////
	nrest_wins = numWindows(rest, FS, 500, 200);
	nfist_wins = numWindows(fist, FS, 500, 200);
	nwavein_wins = numWindows(wavein, FS, 500, 200);
	nwaveout_wins = numWindows(waveout, FS, 500, 200);
	nthumbsup_wins = numWindows(thumbsup, FS, 500, 200);
	rest_feats = calculateFeatures(rest, nrest_wins, 500, 200, FS);
	fist_feats = calculateFeatures(fist, nfist_wins, 500, 200, FS);
	wavein_feats = calculateFeatures(wavein, nwavein_wins, 500, 200, FS);
	waveout_feats = calculateFeatures(waveout, nwaveout_wins, 500, 200, FS);
	thumbsup_feats = calculateFeatures(thumbsup, nthumbsup_wins, 500, 200, FS);
	cout << "Done calculating" << endl;
	cout << "saving" << endl;
	rest_feats.save("rest.dat", raw_ascii);
	fist_feats.save("fist.dat", raw_ascii);
	wavein_feats.save("wavein.dat", raw_ascii);
	waveout_feats.save("waveout.dat", raw_ascii);
	thumbsup_feats.save("thumbsup.dat", raw_ascii);
    //create training set:
	mat rest_train, fist_train, waveout_train, wavein_train, thumbsup_train;
	mat rest_test, fist_test, waveout_test, wavein_test, thumbsup_test;
	mat rest_trainlabels, fist_trainlabels, waveout_trainlabels, wavein_trainlabels, thumbsup_trainlabels;
	mat rest_testlabels, fist_testlabels, waveout_testlabels, wavein_testlabels, thumbsup_testlabels;

	rest_train = rest_feats.rows(0, floor(nrest_wins * 7/ 10));
	rest_test = rest_feats.rows(floor(nrest_wins * 7 / 10)+1, nrest_wins-1);
	rest_trainlabels.ones(floor(nrest_wins * 7 / 10) + 1, 1);
	rest_testlabels.ones(nrest_wins - floor(nrest_wins * 7 / 10) - 1, 1);
	rest_trainlabels = rest_trainlabels * REST;
	rest_testlabels = rest_testlabels * REST;
	//rest_trainlabels.save("rlabels.dat", raw_ascii);
	//rest_testlabels.save("rtlabels.dat", raw_ascii);

	fist_train = fist_feats.rows(0, floor(nfist_wins * 7 / 10));
	fist_test = fist_feats.rows(floor(nfist_wins * 7 / 10) + 1, nfist_wins - 1);
	fist_trainlabels.ones(floor(nfist_wins * 7 / 10) + 1, 1);
	fist_testlabels.ones(nfist_wins - floor(nfist_wins * 7 / 10) - 1, 1);
	fist_trainlabels = fist_trainlabels * FIST;
	fist_testlabels = fist_testlabels * FIST;

	wavein_train = wavein_feats.rows(0, floor(nwavein_wins * 7 / 10));
	wavein_test = wavein_feats.rows(floor(nwavein_wins * 7 / 10) + 1, nwavein_wins - 1);
	wavein_trainlabels.ones(floor(nwavein_wins * 7 / 10) + 1, 1);
	wavein_testlabels.ones(nwavein_wins - floor(nwavein_wins * 7 / 10) - 1, 1);
	wavein_trainlabels = wavein_trainlabels * WAVE_IN;
	wavein_testlabels = wavein_testlabels * WAVE_IN;

	waveout_train = waveout_feats.rows(0, floor(nwaveout_wins * 7 / 10));
	waveout_test = waveout_feats.rows(floor(nwaveout_wins * 7 / 10) + 1, nwaveout_wins - 1);
	waveout_trainlabels.ones(floor(nwaveout_wins * 7 / 10) + 1, 1);
	waveout_testlabels.ones(nwaveout_wins - floor(nwaveout_wins * 7 / 10) - 1, 1);
	waveout_trainlabels = waveout_trainlabels * WAVE_OUT;
	waveout_testlabels = waveout_testlabels * WAVE_OUT;

	thumbsup_train = thumbsup_feats.rows(0, floor(nthumbsup_wins * 7 / 10));
	thumbsup_test = thumbsup_feats.rows(floor(nthumbsup_wins * 7 / 10) + 1, nthumbsup_wins - 1);
	thumbsup_trainlabels.ones(floor(nthumbsup_wins * 7 / 10) + 1, 1);
	thumbsup_testlabels.ones(nthumbsup_wins - floor(nthumbsup_wins * 7 / 10) - 1, 1);
	thumbsup_trainlabels = thumbsup_trainlabels * THUMBS_UP;
	thumbsup_testlabels = thumbsup_testlabels * THUMBS_UP;

	//Create single training matrix
	mat trainmat;
	mat testmat;
	trainmat = join_vert(rest_train,join_vert(fist_train,join_vert(wavein_train,join_vert(waveout_train, thumbsup_train))));
	testmat = join_vert(rest_test, join_vert(fist_test, join_vert(wavein_test, join_vert(waveout_test, thumbsup_test))));

	mat trainlabels;
	mat testlabels;
	trainlabels = join_vert(rest_trainlabels,join_vert(fist_trainlabels,join_vert(wavein_trainlabels,join_vert(waveout_trainlabels, thumbsup_trainlabels))));
	testlabels = join_vert(rest_testlabels, join_vert(fist_testlabels, join_vert(wavein_testlabels, join_vert(waveout_testlabels, thumbsup_testlabels))));

	rowvec trainlabels_vec = vectorise(trainlabels,1);
	rowvec testlabels_vec = vectorise(testlabels,1);

	stdvecvec trainmat_vec, testmat_vec;
	double** trainmat_array;
	double** testmat_array;
	trainmat_vec = mat2StdVec(trainmat);
	testmat_vec = mat2StdVec(testmat);
	trainmat_array = vec2Array2D(trainmat_vec, trainmat.n_rows, trainmat.n_cols);
	testmat_array = vec2Array2D(testmat_vec, testmat.n_rows, testmat.n_cols);
	//cout << trainmat_array[0][0] << endl;
	
	stdvec trainlabels_std = conv_to< stdvec >::from(trainlabels_vec);
	stdvec testlabels_std = conv_to< stdvec >::from(testlabels_vec);

	double* trainlabels_array;
	double* testlabels_array;

	trainlabels_array = &trainlabels_std[0]; //vec2Array(trainlabels_std);
	testlabels_array = &testlabels_std[0];   //vec2Array(testlabels_std);

	//cout << trainlabels_std[25] << endl;
	//cout << "LABELS" << endl;
	//cout << trainlabels_array[25] << endl;
	//cout << testlabels_array[27] << endl;
	cout << trainmat_array[0][0] << endl;
	cout << trainlabels_array[0] << endl;
	//cv::Mat training_data_mat(trainmat.n_rows, trainmat.n_cols, CV_32FC2, trainmat_array);
	//cv::Mat training_data_labels(trainlabels.n_rows, 1, CV_32SC1, trainlabels_array);
	cv::Mat training_data_mat = cv::Mat::zeros(trainmat.n_rows, trainmat.n_cols,CV_32FC1);
	cv::Mat training_data_labels = cv::Mat::zeros(1, trainmat.n_rows, CV_32FC1);

	//populate the matrix
	for (int i = 0; i < trainmat.n_rows;++i) {
		training_data_labels.at<float>(0, i) = trainlabels(i, 0);
		for (int j = 0; j < trainmat.n_cols;++j) {
			training_data_mat.at<float>(i, j) = trainmat(i, j);
		}
	}

	//Print A few values to check
	cout << "Trainvals: " << training_data_mat.at<float>(2, 2) << endl;
	cout << "Trainlabels: " << training_data_labels.at<float>(0, 73) << endl;

	cv::Mat test_data_mat = cv::Mat::zeros(testmat.n_rows, testmat.n_cols, CV_32FC1);
	cv::Mat test_data_labels = cv::Mat::zeros(1, testmat.n_rows, CV_32FC1);
	for (int m = 0; m < testmat.n_rows; ++m) {
		test_data_labels.at<float>(0, m) = testlabels(m, 0);
		for (int n = 0; n < testmat.n_cols; ++n) {
			test_data_mat.at<float>(m, n) = testmat(m, n);
		}
	}

	//Print out sample values to check if Mat is correct
	cout << "Testvals: " << test_data_mat.at<float>(2, 2) << endl;
	cout << "Testlabels: " << test_data_labels.at<float>(0, 20) << endl;
	//Setup SVM Parameters

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	//Train SVM
	CvSVM SVM;
	SVM.train(training_data_mat, training_data_labels, cv::Mat(), cv::Mat(), params);
	//
	
	for (int k = 0; k < test_data_mat.rows; ++k) {
		cv::Mat sample = (Mat_<float>(1, 3) << testmat(k,0),testmat(k,1),testmat(k,2));
		//cout << sample.at<float>(0, 1) << endl;
		//cout << "I'm here" << endl;
		float result = SVM.predict(sample);
		//Print out the predicted class (number btwn 1&5)
		cout << result << endl;
	}
	

	//////////////////////////// End Feature Extraction /////////////////////////////////

	int state = INIT;
	int sample_count = 0;
	//const int sample_values = WIN_SIZE*FS / 1000;
	const int sample_values = 500 * 200 / 1000; //500ms*200Hz/1000
	double raw_sample[sample_values][8];
	stdvecvec buffer;
    while (1) {
        // In each iteration of our main loop, we run the Myo event loop for a set number of milliseconds.
        // In this case, we wish to update our display 50 times a second, so we run for 1000/20 milliseconds.
        hub.run(1000/FS);
        // After processing events, we call the print() member function we defined above to print out the values we've
        // obtained from any events that have occurred.
		switch (state) { //Begine State Machine

		case INIT:
			cout << "What would you like to do?" << endl;
			cout << "1: Log Gesture Data" << endl << "2: Predict Gestures" << endl;
			cout << "Input: " << endl;
			int choice;
			cin >> choice;
			int gesture_type;
			if (choice == 1) {
				cout << "Enter the gesture you want to log" << endl;
				int i;
				i = 0;
				for (i = 0; i < gestures.size(); i++) {
					cout << i + 1 << ": " << gestures[i] << endl;
				}
				cin >> gesture_type;
				state = LOG;
			}
			else if(choice == 2) {
				state = CLASSIFY;
				//state = DEBUG;
			}
			break;
		case DEBUG:

			break;
		case LOG:
			time_t tick;
			tick = time(0);
			collector.openFiles(gestures[gesture_type-1]);
			while(difftime(time(0),tick) < 5){
			//collector.print();
			if (difftime(time(0), tick) >= 5) {
				//collector.closeFiles();
				state = CLASSIFY;
				break;
			}
			}
			break;

		case FILTER:

			break;

		case EXTRACT_FEATURES: 

		
			break;



		case CLASSIFY:
			/*
			stdvec samples = { collector.emgSamples[0], collector.emgSamples[1], ... }; 
			buffer.push_back(samples);
			if (sample_count == sample_values) {
				mat features = calculateFeatures()
				cv::Mat test_sample(features);
				float label = SVM.predict(test_sample);

				//
				if (label == REST) {
					gesture = REST;
				}else if(label == FIST)
					gesture = FIST
					+
			}
			*/
			break;

		case SEND2SOCKET:

			break;


		} //End state Machine
			
    }

    // If a standard exception occurred, we print out its message and exit.
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Press enter to continue.";
        std::cin.ignore();
        return 1;
    }
}
