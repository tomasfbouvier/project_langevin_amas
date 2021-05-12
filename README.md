# project_langevin_amas
Final project for the subject Advanced methods in applied statistics (KU) 2021. 

For this project we trained a neural network to compute the initial position of electron clouds within a simulated time projection chamber.  The neural network was then compared to test data. Our testing showed decent resultsfor  predicting x and z positions  of  the  initial  electron clouds,  whereas the prediction for the y position were significantly worse.

Contributors:
  - Tomás Fernández Bouvier (@tomasfbouvier)

## Files:
  - simulacion_langevin.py : Implementation of the langevin equation solution in the Time Projection Chamber with the purpose of generating sintetic data for the project purposes
  - Project.py : Algorithm for training a secuential model to predict the starting position of the electron clouds before the migration to the chamber
  - particle_traj : Algorithm to detect peaks of electrons on the pad plance of the TPC, call the algorithm to predict their original positions and recreate the trajectory of an energetic particle through the chamber
