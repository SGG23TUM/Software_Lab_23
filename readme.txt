Lets go through step by step what we have and which order are they work:

1-Object Detection Folder
	We built our own code architecture to run faster rcnn model.
	It randomly split dataset as test (%20) and train (%80). Dataset includes images and json files(contains object labels and bounding box img. coordinates for each objects)
	We feed the model with json files and images that you sent. If you want to run the model, you need to add images to the "data" folder still. I remove images from the zip file because it occupies too much storage.


2- After this object detection step we added relationsips to the json file. We did it by utilizing python scripts as you can see in the "Relationship Annotation" folder. 
We think about what kind of relationship pairs we have, to illustrate the pair of "lighting" and "ceiling" has "mounted on" relationships. And this python code basically uses those "relationship_rules" (Which were defined at almost top of the code), then it "classifies" or in other words "adds" those triplets to the json files. It doesn't add their labels it adds their "Id" to the file, id catalog could be found in "dict.json". At the next step "Relationship Detection", we assume these relationships that we made at this step are ground truth annotations.



3- As third step, we built that machine learning model "RandomForest" that we talked on the meeting.
	To sum up, it is feeded by annotated relationship json files(that we created at 2nd step, and it assumes these are ground truth), then it is training on them, and finally it predicts relationship on testing data.
Furthermore, it creates new json files with predicted relationships,of course it makes some mistakes but we ignore those mistakes.


4- Scene Graph Generation
	In "SceneGraph" folder there is an main.py script. It basically reads json files that we created at the end of 3rd step and print out those triplets on an empty images. 