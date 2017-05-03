/*
 * ELM.hpp
 *
 *  Created on: Mar 19, 2017
 *      Author: cg8327
 */

#ifndef ELM_HPP
#define ELM_HPP

void Train_ELM(double* InputMatrix, unsigned int NumberOfSamples, unsigned int NumberOfFeatures, double* OutputMatrix, unsigned int NumberofOutputFeatures, unsigned int NumberOfHiddenNeurons);

void Predict_ELM(double* InputTestSample, unsigned int NumberOfFeatures, double* PredictedOutput, unsigned int NumberofOutputFeatures);


#endif /* ELM_HPP_ */
