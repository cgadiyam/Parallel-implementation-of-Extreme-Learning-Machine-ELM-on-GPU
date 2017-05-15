Usage:
1) Copy the executable to local file system of Jetson.
2) Copy the data set file to the folder in which the executable is copied to.
3) Open the terminal, go to the file path of the executable and run it.
4) Provide input for number of hidden neurons.
5) The application displays the training computation time, prediction time for each sample and testing Root Mean Squared Error.



Note:
--> The application works for three data sets: Auto-MPG, Energy Efficiency, Parkinson's Telemonitoring
--> To change the data set, macro defined in the main.cpp file has to be changed accodingly and parser funtion call in the main function has to be modified.
--> The application works for other data sets too, but a parser function needs to be added for any new data set, to read the data set values from the data set file according to the format defined.
--> To edit the code and build the executable, nsight eclipse IDE should be used.