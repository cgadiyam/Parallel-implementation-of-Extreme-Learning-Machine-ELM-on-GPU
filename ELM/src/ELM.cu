#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cublas_v2.h>

#include "ELM.hpp"



#define CUDA_CALL(ErrorCode) if(ErrorCode != 0){cout<<"Cuda call failed! Error: "<<ErrorCode<<" at line: "<<__LINE__<<endl;}
#define CURAND_CALL(ErrorCode) if(ErrorCode != 0){cout<<"CuRand call failed! Error: "<<ErrorCode<<" at line: "<<__LINE__<<endl;}
#define CUBLAS_CALL(ErrorCode) if(ErrorCode != CUBLAS_STATUS_SUCCESS){cout<<"CuBlas call failed! Error: "<<ErrorCode<<" at line: "<<__LINE__<<endl;}
#define CUSOLVER_CALL(ErrorCode) if(ErrorCode != 0){cout<<"CuSolver call failed! Error: "<<ErrorCode<<" at line: "<<__LINE__<<endl;}

const int TILE_SIZE = 16;


double *InputLayerWeights, *InputLayerBias, *HiddenlayerWeights;
double *InputSampleMin, *InputSampleMax;
int NumofHiddenNeurons;

using namespace std;

__global__ void Initialize_HiddenLayerOutputMatrix(double* devInputLayerBias, unsigned int NumberOfSamples, unsigned int NumberOfHiddenNeurons, double* HiddenLayerOutputMatrix)
{
	int tx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y*blockDim.y) + threadIdx.y;
	int tID = (ty*NumberOfHiddenNeurons) + tx;
	if (tID < (NumberOfSamples * NumberOfHiddenNeurons))
	{
		HiddenLayerOutputMatrix[tID] = devInputLayerBias[(tID/NumberOfSamples)];
	}
}

__global__ void FillDiagonalMatrixInverse(double *d_SI, double* d_S, unsigned int m, unsigned int n)
{
	int tx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y*blockDim.y) + threadIdx.y;
	int tID = (ty*n) + tx;
	if(tID < (m * n))
	{
		if (tx == ty)
		{
			d_SI[tID] = 1 / d_S[tx];
		}
		else
		{
			d_SI[tID] = 0;
		}
	}
}

__global__ void ScaleInputSample(double *InputVector, double *OutputVector, unsigned int NumofSamples, unsigned int NumofFeatures, double *InputSampleMin, double *InputSampleMax)
{
	int tx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int ty = (blockIdx.y*blockDim.y) + threadIdx.y;
	int tID = (ty*NumofFeatures) + tx;
	if(tID<(NumofSamples * NumofFeatures))
	{
		if(InputSampleMax[tID%NumofFeatures] != InputSampleMin[tID%NumofFeatures])
		{
			OutputVector[tID] = (2*((double)(InputVector[tID] - InputSampleMin[tID%NumofFeatures])/(double)(InputSampleMax[tID%NumofFeatures] - InputSampleMin[tID%NumofFeatures]))) - 1;
		}
		else
		{
			OutputVector[tID] = 0;
		}
	}
}

__global__ void TanH_Kernel(double *InputVector, int VectorLength)
{
	int tID = blockDim.x * blockIdx.x + threadIdx.x;
	if(tID < VectorLength)
	{
		InputVector[tID] = (2.0 / (1.0 + exp (-2*InputVector[tID]))) - 1;
	}
}

__global__ void Sigmoid_Kernel(double *InputVector, int VectorLength)
{
	int tID = blockDim.x * blockIdx.x + threadIdx.x;
	if(tID < VectorLength)
	{
		InputVector[tID] = 1.0 / (1.0 + exp (InputVector[tID]));
	}
}

void PrintArray(double *Arr, int Size)
{
	for(int i=0;i<Size;i++)
	{
		cout<<Arr[i]<<"\t";
	}
	cout<<endl<<endl<<endl;
}

void NormalizeTrainSamples(double *hostInputSample, double *InputSample, double *NormalizedSample, unsigned int NumofSamples, unsigned int NumofFeatures)
{
	//cout<<"In NormalizeTrainSamples"<<endl;
	cudaError_t ErrorCode;
	cublasStatus_t BlasErrorCode;
	cublasHandle_t handle;
	BlasErrorCode = cublasCreate(&handle);
	CUBLAS_CALL(BlasErrorCode);
	cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
	cublasSetPointerMode(handle, mode);
	double *hostInputSampleMin, *hostInputSampleMax;
	hostInputSampleMin = new double[NumofFeatures];
	hostInputSampleMax = new double[NumofFeatures];
	cudaMalloc((void**)&InputSampleMin, NumofFeatures * sizeof(double));
	cudaMalloc((void**)&InputSampleMax, NumofFeatures * sizeof(double));
	for(int i=0;i<NumofFeatures;i++)
	{
		//cout<<"inside first for loop"<<endl;
		//cublasIsamin(handle, ((NumofSamples*NumofFeatures)-i), InputSample+i, NumofFeatures, &InputSampleMin[i]);
		//cublasIsamax(handle, ((NumofSamples*NumofFeatures)-i), InputSample+i, NumofFeatures, &InputSampleMax[i]);
		hostInputSampleMin[i] = hostInputSample[i];
		hostInputSampleMax[i] = hostInputSample[i];
		for(int j=0;j<NumofSamples;j++)
		{
			//cout<<"inside second for loop"<<endl;
			if(hostInputSample[(j*NumofFeatures)+i] < hostInputSampleMin[i])
			{
				hostInputSampleMin[i] = hostInputSample[(j*NumofFeatures)+i];
			}
			if(hostInputSample[(j*NumofFeatures)+i] > hostInputSampleMax[i])
			{
				hostInputSampleMax[i] = hostInputSample[(j*NumofFeatures)+i];
			}
		}
	}
	//cout<<"after min max for loops"<<endl;
	cudaMemcpy(InputSampleMin, hostInputSampleMin, NumofFeatures * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(InputSampleMax, hostInputSampleMax, NumofFeatures * sizeof(double), cudaMemcpyHostToDevice);
	dim3 dimBlock1(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid1((int)ceil((float)NumofFeatures / (float)TILE_SIZE), (int)ceil((float)NumofSamples / (float)TILE_SIZE));
	ScaleInputSample<<<dimGrid1, dimBlock1>>>(InputSample, NormalizedSample, NumofSamples, NumofFeatures, InputSampleMin, InputSampleMax);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"ScaleInputSample Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}
}

void Train_ELM(double* InputMatrix, unsigned int NumberOfSamples, unsigned int NumberOfFeatures, double* OutputMatrix, unsigned int NumberofOutputFeatures, unsigned int NumberOfHiddenNeurons)
{
	//cout<<"In Train_ELM func..."<<endl;

	cudaError_t ErrorCode;
	curandStatus_t RandErrorCode;
	cublasStatus_t BlasErrorCode;
	cusolverStatus_t SolverErrorCode;
	double *devInputMatrix, *devInputMatrix1, *devOutputMatrix, *HiddenLayerOutputMatrix;
	/*float *hostInputSampleMin, *hostInputSampleMax;
	hostInputSampleMin = new float[NumberOfFeatures];
	hostInputSampleMax = new float[NumberOfFeatures];
	float *hostInputMatrix, *hostInputLayerBias, *hostInputLayerWeights, *hostHiddenLayerOutputMatrix, *hostHiddenLayerOutputMatrixInv, *hostHiddenlayerWeights, *h_S, *h_U, *h_V, *h_SI;
	hostInputMatrix = new float[NumberOfSamples*NumberOfFeatures];
	hostInputLayerBias = new float[NumberOfHiddenNeurons];
	hostInputLayerWeights = new float[NumberOfHiddenNeurons*NumberOfFeatures];
	hostHiddenLayerOutputMatrix = new float[NumberOfHiddenNeurons*NumberOfSamples];
	hostHiddenLayerOutputMatrixInv = new float[NumberOfHiddenNeurons*NumberOfSamples];
	hostHiddenlayerWeights = new float[NumberOfHiddenNeurons];
	h_S = new float[NumberOfHiddenNeurons];
	h_U = new float[NumberOfSamples*NumberOfSamples];
	h_V = new float[NumberOfHiddenNeurons*NumberOfHiddenNeurons];
	h_SI = new float[NumberOfSamples*NumberOfHiddenNeurons];*/
	int bytes = NumberOfSamples * NumberOfFeatures * sizeof(double);

	NumofHiddenNeurons = NumberOfHiddenNeurons;

	ErrorCode = cudaMalloc((void**)&devInputMatrix1, bytes);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"devInputMatrix1 CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	ErrorCode = cudaMalloc((void**)&devInputMatrix, bytes);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"devInputMatrix CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	ErrorCode = cudaMemcpy(devInputMatrix1, InputMatrix, bytes, cudaMemcpyHostToDevice);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"InputMatrix to devInputMatrix1 cudaMemcpy failed ! Error Code: "<<ErrorCode<<endl;
	}
	ErrorCode = cudaMalloc((void**)&devOutputMatrix, NumberOfSamples * NumberofOutputFeatures * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"devOutputMatrix CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	ErrorCode = cudaMemcpy(devOutputMatrix, OutputMatrix, NumberOfSamples * NumberofOutputFeatures * sizeof(double), cudaMemcpyHostToDevice);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"OutputMatrix to devOutputMatrix cudaMemcpy failed ! Error Code: "<<ErrorCode<<endl;
	}


	NormalizeTrainSamples(InputMatrix, devInputMatrix1, devInputMatrix, NumberOfSamples, NumberOfFeatures);

	//cout<<"after NormalizeTrainSamples func call"<<endl;

	//To debug
	/*cudaMemcpy(hostInputSampleMin, InputSampleMin, NumberOfFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cout<<"Input sample min for each feature: ";
	for(int i=0;i<NumberOfFeatures;i++)
	{
		cout<<hostInputSampleMin[i]<<"\t";
	}
	cout<<endl<<endl;*/

	//To debug
	/*cudaMemcpy(hostInputSampleMax, InputSampleMax, NumberOfFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cout<<"Input sample max for each feature: ";
	for(int i=0;i<NumberOfFeatures;i++)
	{
		cout<<hostInputSampleMax[i]<<"\t";
	}
	cout<<endl<<endl;*/

	//To debug
	//cudaMemcpy(hostInputMatrix, devInputMatrix, NumberOfSamples * NumberOfFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Training Samples after normalization: ";
	//PrintArray(hostInputMatrix, NumberOfSamples * NumberOfFeatures);

	bytes = NumberOfSamples * NumberOfHiddenNeurons * sizeof(double);
	ErrorCode = cudaMalloc((void**)&HiddenLayerOutputMatrix, bytes);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"HiddenLayerOutputMatrix CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}

	curandGenerator_t gen;
	size_t n = (NumberOfHiddenNeurons*NumberOfFeatures);

	/* Allocate n floats on device */
	ErrorCode = cudaMalloc((void **)&InputLayerWeights, n * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"InputLayerWeights CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	/* Create pseudo-random number generator */
	RandErrorCode = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_CALL(RandErrorCode);
	/* Set seed */
	RandErrorCode = curandSetPseudoRandomGeneratorSeed(gen, 1ULL);
	CURAND_CALL(RandErrorCode);
	/* Generate n floats on device */
	RandErrorCode = curandGenerateLogNormalDouble(gen, InputLayerWeights, n, 0, 0.5);
	CURAND_CALL(RandErrorCode);

	cudaThreadSynchronize();

	//To debug
	//cudaMemcpy(hostInputLayerWeights, InputLayerWeights, NumberOfHiddenNeurons * NumberOfFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Input Layer weights: ";
	//PrintArray(hostInputLayerWeights, NumberOfHiddenNeurons * NumberOfFeatures);

	n = NumberOfHiddenNeurons;
	/* Allocate n floats on device */
	ErrorCode = cudaMalloc((void **)&InputLayerBias, n * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"InputLayerBias CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	/* Set seed */
	RandErrorCode = curandSetPseudoRandomGeneratorSeed(gen, 2ULL);
	CURAND_CALL(RandErrorCode);
	/* Generate n floats on device */
	RandErrorCode = curandGenerateLogNormalDouble(gen, InputLayerBias, n, 0, 0.5);
	CURAND_CALL(RandErrorCode);

	//To debug
	//cudaMemcpy(hostInputLayerBias, InputLayerBias, NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Input Layer Bias: ";
	//PrintArray(hostInputLayerBias, NumberOfHiddenNeurons);

	// Specify the size of the grid and the size of the block
	dim3 dimBlock1(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid1((int)ceil((float)NumberOfHiddenNeurons / (float)TILE_SIZE), (int)ceil((float)NumberOfSamples / (float)TILE_SIZE));
	//Initialize hidden layer output matrix with input layer bias
	Initialize_HiddenLayerOutputMatrix<<<dimGrid1, dimBlock1>>>(InputLayerBias, NumberOfSamples, NumberOfHiddenNeurons, HiddenLayerOutputMatrix);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"Initialize_HiddenLayerOutputMatrix Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}

	//To debug
	//cudaMemcpy(hostHiddenLayerOutputMatrix, HiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden Layer Output Matrix(after initializing with input layer bias): ";
	//PrintArray(hostHiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples);

	int lda = NumberOfFeatures, ldb = NumberOfFeatures, ldc = NumberOfSamples;
	const double alf1 = 1;
	const double bet1 = 1;
	const double *alpha1 = &alf1;
	const double *beta1 = &bet1;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	BlasErrorCode = cublasCreate(&handle);
	CUBLAS_CALL(BlasErrorCode);

	//cout<<"Sgemm 1st instance"<<endl;

	// Do the actual multiplication using cuBLAS func
	BlasErrorCode = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NumberOfSamples, NumberOfHiddenNeurons, NumberOfFeatures, alpha1, devInputMatrix, lda, InputLayerWeights, ldb, beta1, HiddenLayerOutputMatrix, ldc);
	CUBLAS_CALL(BlasErrorCode);
	
	//To debug
	//cudaMemcpy(hostHiddenLayerOutputMatrix, HiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden Layer Output Matrix(after multiplication with input layer weights): ";
	//PrintArray(hostHiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples);


	dim3 dimBlock3(TILE_SIZE*TILE_SIZE, 1);
	dim3 dimGrid3((int)ceil((float)(NumberOfHiddenNeurons*NumberOfSamples) / (float)(TILE_SIZE*TILE_SIZE)), 1);
	Sigmoid_Kernel<<<dimGrid3, dimBlock3>>>(HiddenLayerOutputMatrix, NumberOfSamples * NumberOfHiddenNeurons);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"Initialize_HiddenLayerOutputMatrix Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}

	//To debug
	//cudaMemcpy(hostHiddenLayerOutputMatrix, HiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden Layer Output Matrix(after sigmoid kernel): ";
	//PrintArray(hostHiddenLayerOutputMatrix, NumberOfHiddenNeurons * NumberOfSamples);

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;
	ErrorCode = cudaMalloc((void **)&devInfo, sizeof(int));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"devInfo CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	SolverErrorCode = cusolverDnCreate(&solver_handle);
	CUSOLVER_CALL(SolverErrorCode);

	// --- device side SVD workspace and matrices
	double *d_U;
	ErrorCode = cudaMalloc((void **)&d_U, NumberOfSamples * NumberOfSamples * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"d_U CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	double *d_V;
	ErrorCode = cudaMalloc((void **)&d_V, NumberOfHiddenNeurons * NumberOfHiddenNeurons * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"d_V CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	double *d_S;
	ErrorCode = cudaMalloc((void **)&d_S, min(NumberOfSamples, NumberOfHiddenNeurons) * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"d_S CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	double *d_SI;
	ErrorCode = cudaMalloc((void **)&d_SI, NumberOfHiddenNeurons * NumberOfSamples * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"d_SI CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}

	// --- CUDA SVD initialization
	SolverErrorCode = cusolverDnDgesvd_bufferSize(solver_handle, NumberOfSamples, NumberOfHiddenNeurons, &work_size);
	CUSOLVER_CALL(SolverErrorCode);
	double *work;
	cudaMalloc((void**)&work, work_size * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"work CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}

	// --- CUDA SVD execution
	SolverErrorCode = cusolverDnDgesvd(solver_handle, 'A', 'A', NumberOfSamples, NumberOfHiddenNeurons, HiddenLayerOutputMatrix, NumberOfSamples, d_S, d_U, NumberOfSamples, d_V, NumberOfHiddenNeurons, work, work_size, NULL, devInfo);
	CUSOLVER_CALL(SolverErrorCode);
	int devInfo_h = 0;
	ErrorCode = cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if(ErrorCode != cudaSuccess)
	{
		cout<<"devInfo to devInfo_h cudaMemcpy failed ! Error Code: "<<ErrorCode<<endl;
	}
	if (devInfo_h != 0)
	{
		std::cout << "Unsuccessful SVD execution\n\n";
	}

	//To debug
	//cudaMemcpy(h_S, d_S, NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"h_S: ";
	//PrintArray(h_S, NumberOfHiddenNeurons);

	//To debug
	//cudaMemcpy(h_U, d_U, NumberOfSamples * NumberOfSamples * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"h_U: ";
	//PrintArray(h_U, NumberOfSamples * NumberOfSamples);

	//To debug
	//cudaMemcpy(h_V, d_V, NumberOfHiddenNeurons * NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"h_V: ";
	//PrintArray(h_V, NumberOfHiddenNeurons * NumberOfHiddenNeurons);

	double *HiddenLayerOutputMatrixInv;
	ErrorCode = cudaMalloc((void**)&HiddenlayerWeights, NumberOfHiddenNeurons * NumberofOutputFeatures * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"HiddenlayerWeights CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}
	ErrorCode = cudaMalloc((void**)&HiddenLayerOutputMatrixInv, NumberOfSamples * NumberOfHiddenNeurons * sizeof(double));
	if(ErrorCode != cudaSuccess)
	{
		cout<<"HiddenLayerOutputMatrixInv CudaMalloc failed ! Error Code: "<<ErrorCode<<endl;
	}

	const double alf2 = 1;
	const double bet2 = 0;
	const double *alpha2 = &alf2;
	const double *beta2 = &bet2;
	lda = NumberOfHiddenNeurons;
	ldb = NumberOfSamples;
	ldc = NumberOfHiddenNeurons;

	dim3 dimBlock2(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid2((int)ceil((float)NumberOfSamples/ (float)TILE_SIZE), (int)ceil((float)NumberOfHiddenNeurons / (float)TILE_SIZE));
	FillDiagonalMatrixInverse<<<dimGrid2, dimBlock2>>>(d_SI, d_S, NumberOfHiddenNeurons, NumberOfSamples);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"FillDiagonalMatrixInverse Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}

	//To debug
	//cudaMemcpy(h_SI, d_SI, NumberOfSamples * NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"h_SI: ";
	//PrintArray(h_SI, NumberOfSamples * NumberOfHiddenNeurons);

	//cout<<"Sgemm 2nd instance"<<endl;

	BlasErrorCode = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, NumberOfHiddenNeurons, NumberOfSamples, NumberOfHiddenNeurons, alpha2, d_V, lda, d_SI, ldb, beta2, HiddenLayerOutputMatrixInv, ldc);
	CUBLAS_CALL(BlasErrorCode);

	//To debug
	//cudaMemcpy(hostHiddenLayerOutputMatrixInv, HiddenLayerOutputMatrixInv, NumberOfSamples * NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden Layer Output Matrix Inv(after 1st multiplication): ";
	//PrintArray(hostHiddenLayerOutputMatrixInv, NumberOfSamples * NumberOfHiddenNeurons);

	const double alf3 = 1;
	const double bet3 = 0;
	const double *alpha3 = &alf3;
	const double *beta3 = &bet3;
	lda = NumberOfHiddenNeurons;
	ldb = NumberOfSamples;
	ldc = NumberOfHiddenNeurons;

	//cout<<"Sgemm 3rd instance"<<endl;

	BlasErrorCode = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NumberOfHiddenNeurons, NumberOfSamples, NumberOfSamples, alpha3, HiddenLayerOutputMatrixInv, lda, d_U, ldb, beta3, HiddenLayerOutputMatrixInv, ldc);
	CUBLAS_CALL(BlasErrorCode);

	//To debug
	//cudaMemcpy(hostHiddenLayerOutputMatrixInv, HiddenLayerOutputMatrixInv, NumberOfSamples * NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden Layer Output Matrix Inv(after 2nd multiplication): ";
	//PrintArray(hostHiddenLayerOutputMatrixInv, NumberOfSamples * NumberOfHiddenNeurons);

	const double alf4 = 1;
	const double bet4 = 0;
	const double *alpha4 = &alf4;
	const double *beta4 = &bet4;
	lda = NumberOfHiddenNeurons;
	ldb = NumberofOutputFeatures;
	ldc = NumberOfHiddenNeurons;

	//cout<<"Sgemm 4th instance"<<endl;

	BlasErrorCode = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NumberOfHiddenNeurons, NumberofOutputFeatures, NumberOfSamples, alpha4, HiddenLayerOutputMatrixInv, lda, devOutputMatrix, ldb, beta4, HiddenlayerWeights, ldc);
	CUBLAS_CALL(BlasErrorCode);

	//To debug
	//cudaMemcpy(hostHiddenlayerWeights, HiddenlayerWeights, NumberOfHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	//cout<<"Hidden layer Weights: ";
	//PrintArray(hostHiddenlayerWeights, NumberOfHiddenNeurons);

	cudaThreadSynchronize();

	cudaFree(devInputMatrix);
	cudaFree(HiddenLayerOutputMatrix);
	cudaFree(HiddenLayerOutputMatrixInv);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_S);
	cudaFree(d_SI);
	cudaFree(devInfo);
	cudaFree(work);

	BlasErrorCode = cublasDestroy(handle);
	CUBLAS_CALL(BlasErrorCode);

	SolverErrorCode = cusolverDnDestroy(solver_handle);
	CUSOLVER_CALL(SolverErrorCode);

	RandErrorCode = curandDestroyGenerator(gen);
	CURAND_CALL(RandErrorCode);

	//cudaThreadSynchronize();
	//cudaThreadExit();
}

void Predict_ELM(double* InputTestSample, unsigned int NumberOfFeatures, double* PredictedOutput, unsigned int NumberofOutputFeatures)
{
	cudaError_t ErrorCode;
	cublasStatus_t BlasErrorCode;
	//cout<<"In Predict_ELM func..."<<endl;
	double *devInputSample, *devNormalizedInputSample, *HiddenLayerOutput, *devOutput;
	ErrorCode = cudaMalloc((void**)&devInputSample, NumberOfFeatures * sizeof(double));
	CUDA_CALL(ErrorCode);
	ErrorCode = cudaMalloc((void**)&devNormalizedInputSample, NumberOfFeatures * sizeof(double));
	CUDA_CALL(ErrorCode);
	ErrorCode = cudaMalloc((void**)&HiddenLayerOutput, NumofHiddenNeurons * sizeof(double));
	CUDA_CALL(ErrorCode);
	ErrorCode = cudaMalloc((void**)&devOutput, NumberofOutputFeatures * sizeof(double));
	CUDA_CALL(ErrorCode);
	ErrorCode = cudaMemcpy(devInputSample, InputTestSample, NumberOfFeatures * sizeof(double), cudaMemcpyHostToDevice);
	CUDA_CALL(ErrorCode);
	//cout<<"Input sample copied to device.."<<endl;

	dim3 dimBlock1(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid1((int)ceil((float)NumberOfFeatures / (float)TILE_SIZE), 1);
	ScaleInputSample<<<dimGrid1, dimBlock1>>>(devInputSample, devNormalizedInputSample, 1, NumberOfFeatures, InputSampleMin, InputSampleMax);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"ScaleInputSample Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}

	cublasHandle_t handle;
	BlasErrorCode = cublasCreate(&handle);
	CUBLAS_CALL(BlasErrorCode);

	BlasErrorCode = cublasDcopy( handle, NumofHiddenNeurons, InputLayerBias, 1, HiddenLayerOutput, 1);
	CUBLAS_CALL(BlasErrorCode);

	//cout<<"after cublasScopy.."<<endl;

	const double alf = 1;
	const double bet = 1;
	const double *alpha = &alf;
	const double *beta = &bet;

	BlasErrorCode = cublasDgemv( handle, CUBLAS_OP_N, NumofHiddenNeurons, NumberOfFeatures, alpha, InputLayerWeights, NumofHiddenNeurons, devNormalizedInputSample, 1, beta, HiddenLayerOutput, 1);
	CUBLAS_CALL(BlasErrorCode);

	//cout<<"after matrix vector multiplication.."<<endl;

	dim3 dimBlock3(TILE_SIZE*TILE_SIZE, 1);
	dim3 dimGrid3((int)ceil((float)(NumofHiddenNeurons) / (float)(TILE_SIZE*TILE_SIZE)), 1);
	Sigmoid_Kernel<<<dimGrid3, dimBlock3>>>(HiddenLayerOutput, NumofHiddenNeurons);
	cudaThreadSynchronize();
	ErrorCode = cudaGetLastError();
	if(ErrorCode != cudaSuccess)
	{
		cout<<"Initialize_HiddenLayerOutputMatrix Kernel failed ! Error Code: "<<ErrorCode<<endl;
	}

	if(HiddenlayerWeights == NULL)
	{
		cout<<"HiddenlayerWeights is NULL !"<<endl;
	}

	if(devOutput == NULL)
	{
		cout<<"devOutput is NULL !"<<endl;
	}
	if(HiddenLayerOutput == NULL)
	{
		cout<<"HiddenLayerOutput is NULL !"<<endl;
	}

	/*float *hostHiddenLayerOutput, *hostHiddenlayerWeights;
	hostHiddenLayerOutput = new float[NumofHiddenNeurons * sizeof(float)];
	hostHiddenlayerWeights = new float[NumofHiddenNeurons * sizeof(float)];
	cout<<"after malloc.."<<endl;
	cudaMemcpy(hostHiddenLayerOutput, HiddenLayerOutput, NumofHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostHiddenlayerWeights, HiddenlayerWeights, NumofHiddenNeurons * sizeof(float), cudaMemcpyDeviceToHost);
	cout<<"after cuda memcpy.."<<endl;
	cout<<"Hidden Layer Output: "<<endl;
	for(int i=0;i<NumofHiddenNeurons;i++)
	{
		cout<<hostHiddenLayerOutput[i]<<"\t";
	}
	cout<<endl<<endl;
	cout<<"Hidden Layer Weights: "<<endl;
	for(int i=0;i<NumofHiddenNeurons;i++)
	{
		cout<<hostHiddenlayerWeights[i]<<"\t";
	}
	cout<<"after for loops.."<<endl;*/

	//BlasErrorCode = cublasDdot(handle, NumofHiddenNeurons, HiddenLayerOutput, 1, HiddenlayerWeights, 1, PredictedOutput);
	//CUBLAS_CALL(BlasErrorCode);

	const double alf2 = 1;
	const double bet2 = 0;
	const double *alpha2 = &alf2;
	const double *beta2 = &bet2;
	int lda = 1, ldb = NumofHiddenNeurons, ldc = 1;

	BlasErrorCode = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, NumberofOutputFeatures, NumofHiddenNeurons, alpha2, HiddenLayerOutput, lda, HiddenlayerWeights, ldb, beta2, devOutput, ldc);
	CUBLAS_CALL(BlasErrorCode);

	ErrorCode = cudaMemcpy(PredictedOutput, devOutput, NumberofOutputFeatures * sizeof(double), cudaMemcpyDeviceToHost);
	CUDA_CALL(ErrorCode);

	//cout<<"after dot product.."<<endl;

	cublasDestroy(handle);
	cudaFree(devInputSample);
	cudaFree(HiddenLayerOutput);
	cudaFree(devOutput);
}

