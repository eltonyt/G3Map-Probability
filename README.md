# G3Map-Probability
## Purpose 
Build Generalized Groundings Graph &amp; calculate probability of human commands based on groundings surrounded

## Environment
- Python: 3.7.4
- Spacy: 3.2.0
- anytree: 2.8.0
- numpy: 1.20.3
- cv2: 3.4.2.16

## How to run:
- python g3.py [videoname] [command]
- i.e. python g3.py test3.mp4 "Put those books on the chair"


## Process
### G3 Map Draft
Used Spacy and generate G3 Map based on dependencies within the input command. G3 Map will be in a tree structure

### Groudings Sensing
Used pretrained YOLOv3 on COCO dataset to sense groundings surrounded.

### G3 Map Update &amp; Probability Update
When new grounding is sensed by the agent, Corresponding Variables within the G3 Map will be updated, Probability of Human Input Commands will be updated. 

## References 
- Thomas Kollar et al. “Generalized Grounding Graphs: A Probabilistic Framework for Understanding Grounded Commands”. In: arXiv:1712.01097 [cs] (Nov. 29, 2017). arXiv: 1712.01097. url: http://arxiv.org/abs/1712.01097
- Record Number of Assembly Line Robots Ordered in 2021. The Great Courses Daily. Nov. 19, 2021. url: https://www.thegreatcoursesdaily.com/record-number-of-assembly-linerobots- ordered-in-2021/
- Rise of the Machines: The Future has Lots of Robots, Few Jobs for Humans — WIRED. url:https://www.wired.com/brandlab/2015/04/rise-machines-future-lots-robots-jobshumans/
- Understanding natural language commands for robotic navigation and. StuDocu. url: https:/ / www . studocu . com / en - us / document / university - of - pennsylvania / integrated -intelligence-for-robotics/understanding-natural-language-commands-for-roboticnavigation-and-mobile-manipulation/726650
- YOLOv3: Real-Time Object Detection Algorithm (What’s New?) viso.ai. Feb. 25, 2021. url:https://viso.ai/deep-learning/yolov3-overview/
