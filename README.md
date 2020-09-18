# Video Chat Emotion Detection

This Repo contains all files related to the creation of this blog post:


This is intended as a proof of concept for the live capture of emotional analytics from your audience while in a video meeting

## Files

### Dataset
* The file is too large to be uploaded to Github directly.
* Instead it can be downloaded here:
* https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge

### realtime_emotion.ipynb
* This is a colab notebook designed to be run on their free GPU's. 
* In this notebook, I build, train, and export the model.

### fer2.json & fer2.h5
* These are the saved model files for the verison I implement.
* You will need this if you plan to run the .py locally.

### haarcascade_frontalface...xml
* Standard haar cascade file, used in the .py file.

### emotion_live_capture.py
* This py file should be run from the terminal of the machine you want to screen capture.
* This will create a window and live analytics of the 7 emotions measured.
* Currently set to 4 max faces, you can change this in the file.
* I tuned the window locations for my machine with two monitors, so placement may be different on a different setup. 
* This will output a comma separated text file of the captured results. 

### emotion_results.txt
* This is a sample report generated from running the .py file.

### emotions_analytics
* This is a short notebook generating some reporting from the results.
* These images can be seen in the blog post mentioned above. 

### Acknowledgements:
* http://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/
* https://medium.com/@jsflo.dev/training-a-tensorflow-model-to-recognize-emotions-a20c3bcd6468
* https://github.com/neha01/Realtime-Emotion-Detection
* https://github.com/makerportal/pylive


