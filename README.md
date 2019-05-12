# Deep-Learning-Web-App
A Flask app with Deep Learning back-end model trained to classify Indian Vehicles on a dataset curated by me and my team. This is a module from our Final Year project - An Automatic Traffic Management System. We curated a dataset of more than 15k Indian vehicles and trained a Deep Convolutional Neural Network for classification purposes.
The model has been trained using Keras and Tensorflow. Model file has been not been provided in the repo, but can be obtained by training the model yourself. The flask app has been created which utilizes this model and a custom code to pass a given image from the web page through the model and display them back to the web page. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
Clone this repo and run jupyter notebook from the directory. Launch Vehicle_Classifier.ipynb to see training and testing code. 
For dataset, contact me.
Pre-Trained model has been provided, with good training accuracy of about 95%, but bad validation (around 60%). 

### Prerequisites

What things you need to install the software and how to install them
A requirements.txt file is provided if you want to duplciate my work environment. Everything runs smoothly with these settings. 


## Running the tests
Run the testing code from the jupyter notebook.


## Deployment

Add additional notes about how to deploy this on a live system
* Install Flask in your work environment
* Set global variable "FLASK_APP" according to your OS to the name of the python file in which the flask code is written.
* Run the Flask App. If you want it to be accessed by all devices in your network, use host as 0.0.0.0
* Access the website using the hosting PC's IP Address obtained using ipconfig command.
* Enter the URL: <your_ip_address>:5000/static/predict.html
* Choose a png image only. 
* Witness MAGIC and UNLIMITED POWERRR!

## Author

* **Sarvesh Patil**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
