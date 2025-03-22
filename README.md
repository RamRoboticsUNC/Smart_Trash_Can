# Smart_Trash_Can

Maria:
Github Link: https://github.com/chungchunwang/Smartsort

Follow the instructions at the bottom of this link. 

Extra Steps I took:
To set up Firebase:
1. Go to Firebase Console
2. Click "Add project" and follow the setup process.
3. Once created, navigate to Build > Realtime Database and Create a Database.
4. Set the stuff in Rules to true like this:
{
  "rules": {
    ".read": true,
    ".write": true
  }
}

5. Go to Realtime Database, Click the "Data" tab, Copy the URL and paste it into main.py where it says: url = 'swap...'
6. In src/firebase.js, replace all the code in it with your config stuff. You can get it all the code at Project Overview -> make an app and then click on it

7. Run: pip install --user torch torchvision fastapi uvicorn requests numpy pillow opencv-python pickle-mixin
8. uvicorn main:app --reload
(for me, I had to uninstall, reinstall, and upgrade a bunch of libraries in order to get it working)

9. Once it works, go to http://127.0.0.1:8000/docs to test it out

10. In main.py: add these print and else statements in these areas for clarity in the console output:

model = inception_v3(pretrained=True)
print("model eval")
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
else: 
    print("gpu not avilalable")


    

@app.post("/")
async def root(weight: str, file: UploadFile = File(...)):
    print("Actually start testing the model")

  with torch.no_grad():
        output = model(input_batch)
        print("you reached the goal")
        print(output)


        

if data[idx] in recyclable:
    recycle = 1
    print(f"The object {data[idx]} is recyclable.")
else:
    print(f"The object {data[idx]} is not recyclable.")


Ezra:
Github link: https://github.com/manuelamc14/waste-classification-model

Follow instructions in this repo's README to install, including changing the
file structure and file paths in `main.py` and `index.py`.

Make sure to set up a virtual environment and run `pip install -r requirements.txt`.

We were running `main.py` instead of `index.py` like the instructions said, so
use `python3 main.py`. You will probably be prompted to download packages even after you
install the requirements.txt, just follow what the errors say.

Once we started getting errors that no longer prompted us to install packages, we started messing with the main.py file. 

You can copy and paste this main.py file, but make sure to keep your correct paths on your machine (lines 14 & 15).

```python
# Dependencies
import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
import tensorflow as tf

# def predict(img_path):
def getPrediction(filename):
     model = tf.keras.models.load_model("/Users/ezraheinberg/Projects/smart_trash_can_project/waste-classification-model/Resources/Model/final_model_weights.hdf5")
     img = load_img('/Users/ezraheinberg/Projects/smart_trash_can_project/waste-classification-model/static/'+filename, target_size=(180, 180))
     img = img_to_array(img)
     img = img / 255
     img = np.expand_dims(img,axis=0)
     #category = model.predict_classes(img)
     category = model.predict_step(img)
     answer = category[0]
     print(answer)
     probability = model.predict(img)
     probability_results = 0

     is_first_greater = tf.math.greater(answer[0], answer[1])
     print(is_first_greater)
     if is_first_greater:
          answer = "Recyclable"
          probability_results = probability[0][1]
     else:
          answer = "Organic"
          probability_results = probability[0][0]

     answer = str(answer)
     probability_results=str(probability_results)

     values = [answer, probability_results, filename]
     return values[0], values[1], values[2]

# print(getPrediction('img1.jpeg'))
```

One limitation is that this model only can tell whether something is recyclable
or organic, it has no feature to distinguish trash.