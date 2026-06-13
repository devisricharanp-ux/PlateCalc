# PlateCalc🍽️ 
This Repository has all the resources to build a Computer vision based model( using YOLOv8) to predict the calories present in a plate

Dataset used - https://www.kaggle.com/datasets/ggrill/foodseg103
1. Dataset Preparation & Model Training
   Use the FoodSeg103 dataset containing food images, segmentation masks, and labels.
   I have used two methods to train the model

##Two-Stage Hierarchical (12-class detector + per-category classifiers)
  
  Pros:
  
  Each stage-2 classifier only needs to distinguish among a small subset (3-20 classes), which is generally easier to train well with limited    data per class
  If some macro-categories have very few images for certain fine-grained classes, the impact is contained to that one classifier rather than     dragging down the whole model
  Easier to debug — if accuracy is poor, you can isolate whether it's a stage-1 (detection/localization) problem or a stage-2 (fine-grained      classification) problem for a specific category
  Can iterate on individual classifiers independently without retraining everything (e.g., if "Fruits" classifier is weak, retrain just that     one)
  Stage-1 detector (12 classes) likely converges faster and more reliably than a 103-class detector, since fewer classes = less confusion in     the detection head
  
  Cons:
  
  Error propagation — if stage 1 misroutes an item to the wrong macro-category, stage 2 has zero chance of correcting it, since that             classifier doesn't even have the correct class as an option
  More moving parts — 13 models total (1 detector + 12 classifiers) means more training runs, more files to manage, more inference latency       (crop → classify, per detected box)
  Crop quality dependency — stage 2 accuracy depends heavily on how good stage-1's bounding boxes are; loose/tight crop mismatches between       training and inference can quietly hurt accuracy
  Imbalanced classifiers — category 7 (20 classes) and category 11 (4 classes, very heterogeneous) will likely perform very differently,         making overall system accuracy uneven
  Slightly more complex deployment/maintenance pipeline

##Single-Stage (103-class YOLOv8 detector)
  
  Pros:
  
  Simpler pipeline — one model, one inference pass, no crop-and-reclassify step, lower latency
  No error propagation between stages — every class is "in the running" for every detection directly
  Easier to deploy and maintain — one model file, one training run
  The detector learns spatial + class features jointly, which can sometimes pick up on contextual cues (e.g., plate position, co-occurring foods) that a cropped classifier loses
  
  Cons:
  
  103-class detection head is harder to train well, especially with limited data per class — visually similar items (different fruits, different leafy greens) are more likely to be confused
  Class imbalance across 103 categories is harder to manage in one model — common classes can dominate and starve rare ones
  You already observed this approach "isn't that great" — likely because the detection head has to learn fine-grained visual distinctions AND localization simultaneously across 103 classes, which is a much harder joint optimization
  Harder to debug — if accuracy is low, it's not obvious whether it's a data problem for a specific class, an architectural capacity issue, or something else
  Less flexible — improving accuracy for one category (say, fruits) requires retraining the entire 103-class model
