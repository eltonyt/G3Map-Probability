# G3Map-Probability
## Purpose 
Build Generalized Groundings Graph &amp; calculate probability of human commands based on groundings surrounded

## Process
### G3 Map Draft
Used Spacy and generate G3 Map based on dependencies within the input command. G3 Map will be in a tree structure

### Groudings Sensing
Used pretrained YOLOv3 on COCO dataset to sense groundings surrounded.

### G3 Map Update &amp; Probability Update
When new grounding is sensed by the agent, Corresponding Variables within the G3 Map will be updated, Probability of Human Input Commands will be updated. 
