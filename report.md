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



## Task 2: Gradient Descent-based IK (the Jacobian Transpose method)

### Task 2.1: Velocity Jacobian & Column Visualization

I implemented `VelocityJacobian()` so that each column  
\[
J_{:,i} = \mathbf{z} \times (p_{\text{end}} - p_i)
\]
gives the instantaneous linear velocity of the end effector when joint \(i\) rotates. To verify correctness, I drew each column as a colored arrow anchored at the end effector:

- **Color scheme:**  
  - Base (joint 0): gray  
  - Joint 1: red  
  - Joint 2: green  
  - Joint 3: blue  
- **Arrow direction:** perpendicular to the vector from joint \(i\) to the end effector, matching the lecture’s “tip-anchored” Jacobian diagram.  
- **Scaling:** arrows are scaled for visibility; their directions exactly match \(\mathbf{z} \times (p_{\text{end}} - p_i)\).

**Screenshot showing all columns at the end effector:**

![Jacobian Columns Visualization](images/jacobian_columns_demo.png)

> When I adjust any joint angle, the corresponding arrow updates immediately, confirming that the Jacobian implementation is correct.


