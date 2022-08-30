from distutils.command.config import config
from app import app
from flask import request, render_template
import cv2
from PIL import Image
from skimage.metrics import structural_similarity
import imutils
import os

#Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['GENERATED_FILES'] = 'app/static/generated'
app.config['EXISTING_FILES'] = 'app/static/original'


#adding the path to config file
app.route('/', methods = ['GET','POST'])
def index():

    #Execute if request is GET
    if request.method == 'GET':
        return render_template('index.html')
    
    #Executes if request is POST
    if request.method == 'POST':

        #Get upload image
        file_upload = request.files['file_upload']
        file_name = file_upload.name

        #Resizing and saving the uploaded image
        uploaded_image = Image.open(file_upload).resize((250,160))
        uploaded_image.save(os.path.join(app.config['INTIAL_FILE_UPLAODS'],'image.jpg'))

        # Resize and save the original image to ensure both uploaded and original matches in size
        original_image = Image.open(os.path.join(app.config['EXISTING_FILES'],'image.jpg')).resize((250,160))
        original_image.save(os.path.join(app.config['EXISTING_FILES'],'image.jpg'))

        #Reading resized images as array using CV2
        original_image =cv2.imread(os.path.join(app.config['EXISTING_FILES'],'image.jpg'))
        uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

        #converting images to greyscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        uploaded_gray = cv2.cvtColor(uploaded_image,cv2.COLOR_RGB2BGR)

        #Calculating structural similirity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full= True)
        diff = (diff*255).astype('unit8')

        #Calculating the threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        #Drawing contours on the image
        for cnt in cnts:
            (x, y, w, h)=cv2.boundingRect(cnt)
            cv2.rectangle(original_image, (x,y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(uploaded_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        #Saving all output images
        cv2.imwrite(os.path.join(app.config('GENERATED_FILES'), 'image_original.jpg'), original_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILES'], 'image_uploaged.jpg'), uploaded_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILES'], 'image_diff.jpg'),diff)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILES'], 'image_thresh.jpg'), thresh)
        return render_template('index.html', pred=str(round(score*100),2)+ '%'+ 'is matching')



#Main function
if __name__ == '__main__':
    app.run(debug=True)