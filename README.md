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

