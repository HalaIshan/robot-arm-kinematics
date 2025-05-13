# Animation and Robotics - Assignment 3: Kinematics

In this assignment, you will implement a simple optimization-based inverse kinematics application for a 2D robotic arm with hinge joints. 

## Instructions

### Preliminary steps:
1. Ensure you have completed the setup steps from Assignment 1, including the installation of Python, VS Code, and required extensions.
2. Make sure you have cloned the repository and have the environment set up correctly.

### Setup steps:
1. Create a folder with no spaces and no non-English characters (Preferably in `C:\Courses\AnimationAndRobotics\Assignments\`) and clone the assignment repository with `git clone`:

    `git clone https://github.com/xCraftCourses/animation-and-robotics-simple-IK-[your github id]`
    
    This will create a new folder that contains this repository.
2. Open the folder with VS Code.
3. Create a new Python environment (`CTRL-SHIFT-P`, type `python env` and select `Python: Create Environment`). Follow the steps. VS Code should create a new folder called `.venv`.
4. Open a new terminal (`` CTRL-SHIFT-` ``). If VS Code detected the Python environment correctly, the prompt should begin with `(.venv)`. If not, restart VS Code and try again. If it still doesn't work, make sure the default terminal is `cmd` or `bash` (use `CTRL-SHIFT-P` and then `Terminal: Select Default Profile` to change it) and start a new terminal. If it still fails, ask for help.
5. Install Vedo, a scientific visualization package for Python, using `pip install vedo` in the terminal.
6. Open `Assignment3.py`. The file is divided into cells, where each cell is defined by `#%%`. Run the first cell. Recall that VS Code will tell you that it needs to install the ipykernel runtime. Make sure it ran without any errors.
7. Run all the cells. A window with a 2D robotic arm should appear.

## Introduction

The initial code for the assignment contains a partial implementation of a simple robotic arm class with identical links. The constructor accepts the number of links and initializes the relevant values. The class implements Forward Kinematics (FK) for the case of 3 joints, which you will be required to extend, and the implementation of Inverse Kinematics (IK) is left empty.

The GUI allows the user to view the arm and change its joint angles. The left slider determines which joint the slider on the right controls. The blue sphere represents the IK target. Clicking anywhere in the window repositions the target.

## Tasks
You are required to write a report in a markdown format (`.md`). The report will be fetched and processed automatically, so it is important you follow these steps:
1. Create an .md file with the work `report` in its name (e.g. `report.md`) and put it in the root path of the repository.
2. Put your full name and ID number somewhere in the report.

The report should be divided into tasks. In each part, use images and videos, with extra text to complement them, to show what you did. The images and videos should be added to the repository. They should clearly and *concisely* demonstrate the tasks were performed correctly. Avoid adding too many images, or videos longer than 5 seconds, as this does not add value to the report and in fact can even make it worse. 

### Task 1: Extend the implementation of the FK function to any number of joints.

The current implementation considers only the case where there are 3 joints and all of the links have identical lengths. 
1. Change `SimpleArm` such that the constructor can accept different link lengths. Set the link length to some arbitrary length of your choice and change the robot pose in the GUI. Add an image to the report.
2. The FK method computes the position of each joint in world coordinates explicitly. Change the code such that the FK can support any number of joints by replacing the computation with a loop. Create an arm with an arbitrary number of joints, pose it, and add the result to the report.

### Task 2: Gradient Descent-based IK (the Jacobian Transpose method)
1. Implement the `VelocityJacobian` method in `SimpleArm`. Check that your implementation is correct by visualizing the columns of the Jacobian. Explain your reasoning in the report.
2. Implement the `IK` method in `SimpleArm` using gradient descent and adapt the GUI accordingly. Try to make your implementation as robust as you can. Explain your implementation choices in the report.
3. Add several images demonstrating IK to your report. In particular, experiment with placing the IK target beyond what the arm can reach. Explain what you see.

### Task 3: Gauss-Newton-based IK (the Jacobian Inverse method)
1. Add the option to use Gauss-Newton instead of gradient descent in `IK` and adapt the GUI accordingly. Pick a target pose and take screenshots of several consecutive iterations and add them to the report. How did the progression of IK change with respect to gradient descent? Explain in the report.
2. Cases where the robot possesses more than 2 joints are said to be *redundant*, meaning there are extra degrees of freedom, and that the IK solution is not unique. Propose and implement an approach to define a unique solution.

### Task 4: Extensions
1. Allow the user to set limit joint angles to a specific range (e.g. from -45 to 45 degrees). Extend the solver with the ability to adhere to these limits. Explain your approach in the report. 
2. The manipulability ellipsoid of a robot at a certain pose is an ellipsoid centered at the end effector, with axis aligned with the singular vectors of the velocity Jacobians, and scaled according to the singular values. Compute and draw the manipulability ellipsoid in the GUI. 
3. Extend your implementation to include prismatic joints.


## Submission
1. Place the report in a file named `report.md` in the root directory of the repository.
2. Push your changes to the GitHub repository. Every subtask in the assignment must have a unique commit with a commit message identifying it as such. For example, once you finish subtask 2.3, create a commit with the message "Subtask 2.3 finished".
3. The last commit before the deadline is the one that will be examined. Later commits will be ignored.

## Grading
Grading will be based on the following criteria:
* Correctness of the results
* Quality of the report
* Creativity
* Effort
* Commit history

The feedback will appear on the feedback pullrequest of your repository.