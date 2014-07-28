This directory contains all of the sigsets used in the Video Challenge Problem (VCP) Version 2.0.

Experiment Types
The VCP consist of three (3) primary experiments type
 1A. Face Walking Video             vs.  Face Walking Video
 2A. Face Walking & Activity Video  vs.  Face Walking & Activity Video
 3A. Face Activity Video            vs.  Face Activity Video

Sigsets
A target and a query sigset are associated with each of the 3 experiments. However,
the target sigsets are divided into multiple sigsets to guarantee that at most one 
image of a given subject is in the sigset. Thus, each experiment consists of matching
multiple target sigsets to a common query sigset. The following table lists the 
target and query sigsets for each of the 3 experiments and the number of individual target
sigsets.

Name         Target Sigsets         #         Query Sigset
 ---  ---------------------------  ---  -------------------------
  1A   face_walking_video_#.xml      4   face_walking_video.xml 
  2A   face_walking_video_#.xml      4   face_activity_video.xml
  2A   face_activity_video_#.xml     4   face_walking_video.xml 
  3A   face_activity_video_#.xml     4   face_activity_video.xml
 
Development Sigsets
A set of development sigsets that list images that can be used for testing and training
is also provided. The images in the development sets are not included in the sigsets used
for the experiments. The development sigsets are listed below.

          Image Type           Exp         Development Sigset
---------------------------   -----  --------------------------------
face_walking_video.xm         1A,2A  dev_face_walking_video.xml
face_activity_video.xml       2A,3A  dev_face_activity_video.xml

*** WARNING ***
---------------
THE SUBJECTS USED IN EXPERIMENTS 1A, 2A AND 3A ARE NOT THE SAME SUBJECTS USED IN
EXPERIMENTS 1, 2 AND 3. THE SUBJECTS IDS FOR EXPERIMENTS 1A-3A ARE OF the FORM
'ud1S#####' from the University of Texas at Dallas WHILE THOSE FOR EXPERIMENTS 1-3
ARE OF THE FORM 'nd1S#####' from the University of Notre Dame. THUS, SUBJECTS 
ud1S04500 AND nd1S04500 ARE NOT THE SAME PERSON!!!