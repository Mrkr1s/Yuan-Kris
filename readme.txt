Project: Deep Learning Assignment 1
-------------------------------------
This project implements a Multi-Layer Perceptron (MLP) using only NumPy to perform a 10-class classification task, 
following the requirements of COMP5329 (no deep learning frameworks allowed).

project/
│
├── main.py               # Main script containing training and evaluation loops  
├── model.py              # Defines the MLP network architecture  
├── layers.py             # Implements core layers (Linear, ReLU, BatchNorm, Dropout)  
├── loss.py               # Implements Softmax + Cross-Entropy loss  
├── optimizer.py          # Implements SGD with Momentum and L2 regularization  
├── data_utils.py         # Data loading, normalization, and mini-batch utilities  
├── train_data.npy        # Training data  
├── train_label.npy       # Training labels  
├── test_data.npy         # Test data  
├── test_label.npy        # Test labels  
├── README.txt            # Instructions  
└── report.pdf            # Full report with experiment details  

How to Use:
-------------
	1.	Make sure all .py files and .npy data files are placed in the same directory.
	2.	Install required Python packages (e.g., numpy, matplotlib):
   pip install numpy matplotlib
	3.	Run the main script in the terminal:
   python main.py
	4.	The script will automatically load the data, train the model, and show loss/accuracy curves.
	5.	For full experimental analysis and results, check the report.pdf.