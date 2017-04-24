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

//Define Classes
#define REST 0x3E8;
#define FIST 0x7D0;
#define WAVE_IN 0xBB8;
#define WAVE_OUT 0xFA0;
#define THUMBS_UP 0x1388;
#define OK 0x1770;
#define OPEN_HAND 0x1B58;
#define PEACE 0x1F40;
#define ROCK_ON 0x1F40;
//Will add more gesture definitions here.


//Set Parameters for the training data
const int num_classes = 5;
const int fs = 200;

class DataCollector : public myo::DeviceListener {
public:
    DataCollector()
    : emgSamples()
    {
		//openFiles();
    }

	void openFiles(string name) {
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

vector<int> createVector(int startval, int stepval, int endval) {
	vector<int> result;
	while (startval <= endval) {
		result.push_back(startval);
		startval += stepval;
	}
	return result;
}
int calculateNumWindows(vector<int> window, int sampling_freq, int win_length, int win_disp) {
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

void printVector(vector<int> vec) {
	for (int i = 0; i < vec.size(); ++i) {
		cout << vec[i] << endl;
	}
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
	int state = INIT;
    while (1) {
        // In each iteration of our main loop, we run the Myo event loop for a set number of milliseconds.
        // In this case, we wish to update our display 50 times a second, so we run for 1000/20 milliseconds.
        hub.run(1000/fs);
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
				state = EXTRACT_FEATURES;
			}
			break;
	    
		case LOG:
			time_t tick, tock;
			tick = time(0);
			collector.openFiles(gestures[gesture_type-1]);
			while(difftime(time(0),tick) < 10){
			collector.print();
			if (difftime(time(0), tick) >= 10) {
				state = INIT;
				break;
			}
			}
			break;

		case FILTER:

			break;

		case EXTRACT_FEATURES: {
			string train_data_file;
			train_data_file = "train_raw.csv";
			mat raw_data;
			raw_data.load(train_data_file, csv_ascii);
			cout << raw_data.n_cols << endl << raw_data.n_rows << endl;
			

		}
			break;

		case CLASSIFY:
			//dummy returned gesture
			int gesture;
			//Run classification routine:

			//Logistic Regression

			//SVM

			//LDA

			gesture = OK;
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
