#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <ctime>
#include "ELM.hpp"

using namespace std;

#define NUMBEROFSAMPLES 5875
#define NUMBEROFTRAINSAMPLES 875
//#define NUMBEROFHIDDENNEURONS 10
#define NUMBEROFFEATURES 16
#define NUMBEROFOUTPUTFEATURES 2


void ParseData1(double* InputSet, double* OutputSet, int NumOfSamples)
{
	ifstream DataSetFile("auto-mpg.txt");
	string line;
	const char *SampleData;
	int index = 0;
	char Data[8][10];

	if (!DataSetFile)
	{
		cout << "couldn't open the file !!!";
	}

	for (int i = 0; i<NumOfSamples; i++)
	{
		if (DataSetFile)
		{
			index = i * 7;
			getline(DataSetFile, line);
			SampleData = line.c_str();
			sscanf(SampleData, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s", Data[0], Data[1], Data[2], Data[3], Data[4], Data[5], Data[6], Data[7]);
			OutputSet[i] = atof(Data[0]);
			for (int j = 0; j<7; j++)
			{
				InputSet[index + j] = atof(Data[j + 1]);
			}
		}
	}

}

void ParseData2(double* InputSet, double* OutputSet, int NumOfSamples)
{
	ifstream DataSetFile("ENB2012_data.csv");
	string line;
	const char *SampleData;
	char Data[10][10];

	if (!DataSetFile)
	{
		cout << "couldn't open the file !!!";
	}

	for (int i = 0; i<NumOfSamples; i++)
	{
		if (DataSetFile)
		{
			getline(DataSetFile, line);
			SampleData = line.c_str();
			sscanf(SampleData, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s", Data[0], Data[1], Data[2], Data[3], Data[4], Data[5], Data[6], Data[7], Data[8], Data[9]);
			for (int j = 0; j<8; j++)
			{
				InputSet[(i * 8) + j] = atof(Data[j]);
			}
			for (int j = 0; j<2; j++)
			{
				OutputSet[(i * 2) + j] = atof(Data[j + 8]);
			}
		}
	}

}

void ParseData3(double* InputSet, double* OutputSet, int NumOfSamples)
{
	ifstream DataSetFile("parkinsons_updrs.csv");
	string line;
	const char *SampleData;
	char Data[22][10];

	if (!DataSetFile)
	{
		cout << "couldn't open the file !!!";
	}

	for (int i = 0; i<NumOfSamples; i++)
	{
		if (DataSetFile)
		{
			getline(DataSetFile, line);
			SampleData = line.c_str();
			sscanf(SampleData, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s", Data[0], Data[1], Data[2], Data[3], Data[4], Data[5], Data[6], Data[7], Data[8], Data[9], Data[10], Data[11], Data[12], Data[13], Data[14], Data[15], Data[16], Data[17], Data[18], Data[19], Data[20], Data[21]);
			for (int j = 0; j<16; j++)
			{
				InputSet[(i * 16) + j] = atof(Data[j + 6]);
			}
			for (int j = 0; j<2; j++)
			{
				OutputSet[(i * 2) + j] = atof(Data[j + 4]);
			}
		}
	}

}


int main()
{
	int Num0fNeurons=0;
	double Train_Time=0, Predict_Time=0;
	clock_t start, end;
	double *InputSet, *OutputSet, *InputTestSample, *ActualOutput, *PredictedOutput, *Error, *TrainInputSet, *TrainOutputSet;
	InputSet = new double[NUMBEROFSAMPLES * NUMBEROFFEATURES];
	OutputSet = new double[NUMBEROFSAMPLES * NUMBEROFOUTPUTFEATURES];
	TrainInputSet = new double[NUMBEROFTRAINSAMPLES * NUMBEROFFEATURES];
	TrainOutputSet = new double[NUMBEROFTRAINSAMPLES * NUMBEROFOUTPUTFEATURES];
	InputTestSample = new double[NUMBEROFFEATURES];
	ActualOutput = new double[NUMBEROFOUTPUTFEATURES];
	PredictedOutput = new double[NUMBEROFOUTPUTFEATURES];
	Error = new double[(NUMBEROFSAMPLES-NUMBEROFTRAINSAMPLES) * NUMBEROFOUTPUTFEATURES];
	ParseData3(InputSet, OutputSet, NUMBEROFSAMPLES);
	memcpy(TrainInputSet, InputSet, NUMBEROFTRAINSAMPLES * NUMBEROFFEATURES * sizeof(double));
	memcpy(TrainOutputSet, OutputSet, NUMBEROFTRAINSAMPLES * NUMBEROFOUTPUTFEATURES * sizeof(double));
	double rmse;
	cout<<"Number of Neurons:\t";
	cin>>Num0fNeurons;
	/*int index = 0;
	for (int i = 0; i < NUMBEROFSAMPLES; i++)
	{
	index = i * 7;
	cout << OutputSet[i];
	for (int j = 0; j < 7; j++)
	{
	cout << '\t' << InputSet[index + j];
	}
	cout << '\n';
	}*/

	start = clock();
	Train_ELM(TrainInputSet, NUMBEROFTRAINSAMPLES, NUMBEROFFEATURES, TrainOutputSet, NUMBEROFOUTPUTFEATURES, Num0fNeurons);
	end = clock();
	Train_Time = (double)(end - start) * 1000 / (double)CLOCKS_PER_SEC;
	cout << "\nTraining Computation time: " << Train_Time << " ms";

	for(int j=0;j<(NUMBEROFSAMPLES-NUMBEROFTRAINSAMPLES);j++)
	{
		for(int i=0;i<NUMBEROFFEATURES;i++)
		{
			InputTestSample[i] = InputSet[i+((NUMBEROFTRAINSAMPLES+j)*NUMBEROFFEATURES)];
		}
		for(int i=0;i<NUMBEROFOUTPUTFEATURES;i++)
		{
			ActualOutput[i] = OutputSet[i+((NUMBEROFTRAINSAMPLES+j)*NUMBEROFOUTPUTFEATURES)];
		}
		//*ActualOutput = OutputSet[NUMBEROFTRAINSAMPLES+j];

		start = clock();
		Predict_ELM(InputTestSample, NUMBEROFFEATURES, PredictedOutput, NUMBEROFOUTPUTFEATURES);
		end = clock();
		Predict_Time += (double)(end - start) * 1000 / (double)CLOCKS_PER_SEC;

		for(int i=0;i<NUMBEROFOUTPUTFEATURES;i++)
		{
			Error[(j*NUMBEROFOUTPUTFEATURES) + i] = PredictedOutput[i] - ActualOutput[i];
		}
		//Error[j] = (*PredictedOutput) - (*ActualOutput);

		/*cout<<"Actual Output:";
		for(int i=0;i<NUMBEROFOUTPUTFEATURES;i++)
		{
			cout<<" "<<ActualOutput[i];
		}
		cout<<"\t"<<"Predicted Output:";
		for(int i=0;i<NUMBEROFOUTPUTFEATURES;i++)
		{
			cout<<" "<<PredictedOutput[i];
		}
		cout<<"\t"<<"Error:";
		for(int i=0;i<NUMBEROFOUTPUTFEATURES;i++)
		{
			cout<<" "<<Error[(j*NUMBEROFOUTPUTFEATURES) + i];
		}
		cout<<endl;*/
	}

	Predict_Time = Predict_Time/(double)(NUMBEROFSAMPLES-NUMBEROFTRAINSAMPLES);
	cout << "\nPrediction Computation time(for each sample): " << Predict_Time << " ms";

	rmse = 0;
	for(int i=0;i<(NUMBEROFSAMPLES-NUMBEROFTRAINSAMPLES)*NUMBEROFOUTPUTFEATURES;i++)
	{
		rmse += (double)(Error[i]*Error[i]);
	}
	rmse = sqrt(rmse)/(double)((NUMBEROFSAMPLES-NUMBEROFTRAINSAMPLES)*NUMBEROFOUTPUTFEATURES);
	cout<<"\nRoot mean squared error: "<<rmse;

	getchar();
	return 0;
}
