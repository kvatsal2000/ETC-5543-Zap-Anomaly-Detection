# ETC-5543-Zap-Anomaly-Detection
This is the GitHub repository for the ETC:5543 Internship Project of Kumar Vatsal and Dhruv Nirmal.

This repository contains the following:

- Presentation and the code to reproduce the plots in it in the **Presentation folder**
- Report and the code to reproduce the analysis and plots in the **Report Folder**
- Code for the anomaly detection process in the **Code Folder**.It contains Local outlier factor in LOF_function.py file, Isolation Forest in IF_function.py file and finally the combinaton and the final output file in the anomaly_detection.py file.

To reproduce the analysis, the user must gain access to the database from VCDI and then load the data in the python scripts. Then Run the anomaly_detection.py file. 
On running this function, the output will be a table that contains the **top 20 anomalies** for every month across all the matrices. 
