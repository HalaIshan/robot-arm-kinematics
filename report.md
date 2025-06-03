# Report for Assignment 3 - Kinematics

**Name:** Hala Ishan  
**Student ID:** 213692296

## Task 1: Extended FK for Arbitrary Link Lengths

### Task 1.1: Support for Arbitrary Link Lengths

The original implementation assumed all links had identical lengths of 1 unit. I modified the `SimpleArm` constructor to accept a list of arbitrary link lengths using a new `link_lengths` parameter. This allows flexible creation of arms with non-uniform segments.

Link lengths chosen for demonstration: `[1, 2, 0.5]`

Resulting robot configuration (screenshot from GUI):

![Custom Link Lengths FK](images/custom_link_lengths_fk.png)

