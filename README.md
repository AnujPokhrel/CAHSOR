1. requirements.txt has all the packages necessary to run this project.

2. DataProcessingPipleline folder contains python scripts necessary to process bags collected before training and deploying.
	->ProcessBags.py script runs the script to process collected bags

3. TRON folder contains the TRON method as discussed in the paper as well as the Downstream Kinodynamic Learning.

4. Manual_CAHSOR contains files to run the Manual + CAHSOR implementation in the robot.
	->Manual_CAHSOR.py script runs the implementation.

5. MPPI_CAHSOR contains files to run the MPPI + CAHSOR implementation in the robot.
	->mppi_planner.py script runs the implementation.

## Cite our paper:
```
@article{pokhrel2024cahsor,
  author={Pokhrel, Anuj and Nazeri, Mohammad and Datar, Aniket and Xiao, Xuesu},
  journal={IEEE Robotics and Automation Letters}, 
  title={CAHSOR: Competence-Aware High-Speed Off-Road Ground Navigation in $\mathbb {SE}(3)$}, 
  year={2024},
  volume={9},
  number={11},
  pages={9653-9660},
  doi={10.1109/LRA.2024.3457369}}
```
