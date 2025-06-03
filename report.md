# Report for Assignment 3 - Kinematics

**Name:** Hala Ishan  
**Student ID:** 213692296

## Task 1: Extended FK for Arbitrary Link Lengths

### Task 1.1: Support for Arbitrary Link Lengths

The original implementation assumed all links had identical lengths of 1 unit. I modified the `SimpleArm` constructor to accept a list of arbitrary link lengths using a new `link_lengths` parameter. This allows flexible creation of arms with non-uniform segments.

Link lengths chosen for demonstration: `[1, 2, 0.5]`

**Resulting robot configuration (screenshot from GUI):**

![Custom Link Lengths FK](images/custom_link_lengths_fk.png)


### Task 1.2: Generalizing FK with a Loop

While implementing Task 1.1, I had already completed the core of Task 1.2 without realizing it (I hadn’t read Task 1.2 until after finishing 1.1). Specifically, I updated the `FK` method to use a loop instead of hardcoded steps. 

This generalization allows the arm to support any number of joints, with joint positions computed dynamically using cumulative rotations.

To demonstrate this, I created a 5-joint arm using the configuration `[1, 0.5, 1.2, 0.5, 2]` and posed it using the GUI.

**Resulting pose with 5 joints:**

![Generalized FK 5-Joint Example](images/fk_five_joints_demo.png)
