# nn_laser_spot_detection
Detect (and practically track) a laser spot emitter from a common laser pointer

## Requirments:
- it should work on cpu only pc, but gpu (only nvidia) is preferrable
- pytorch with matching version of cuda. conda (please miniconda) can be used but I think you have to install also ros in the environent to run all
- ros obviously
- For fake tracking: https://github.com/torydebra/tf_visual_tools.git

## Setup and running

- Indentify the model you want (yolo large is good) from [here](https://istitutoitalianotecnologia-my.sharepoint.com/personal/davide_torielli_iit_it/_layouts/15/onedrive.aspx?login_hint=davide%2Etorielli%40iit%2Eit&id=%2Fsites%2FHHCMResearchLine%2DPhD%5FMuratore%5FTsagarakis%2FDocumenti%20condivisi%2Ftori%5Flaser%5Ftrained%5Fmodels&listurl=https%3A%2F%2Fistitutoitalianotecnologia%2Esharepoint%2Ecom%2Fsites%2FHHCMResearchLine%2DPhD%5FMuratore%5FTsagarakis%2FDocumenti%20condivisi&viewid=104ee7db%2Dbb95%2D4791%2Db54d%2D8f1a1b334672&view=0) (three main ones) or [here](https://istitutoitalianotecnologia-my.sharepoint.com/personal/davide_torielli_iit_it/_layouts/15/onedrive.aspx?login_hint=davide%2Etorielli%40iit%2Eit&id=%2Fpersonal%2Fdavide%5Ftorielli%5Fiit%5Fit%2FDocuments%2FLearningData&view=0) (all versions)

- Check arguments in this launch, and launch it: `roslaunch tpo_vision laser3DTracking.launch`

## Training your how model
- See https://github.com/ADVRHumanoids/nn_laser_spot_tracking/tree/master/training
