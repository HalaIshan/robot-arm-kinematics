#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np

#%% class for a robot arm
def Rot(angle, axis):
    # calculate the rotation matrix for a given angle and axis using Rodrigues' formula
    # return a 3x3 numpy array
    # also see scipy.spatial.transform.Rotation.from_rotvec
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R
    
class SimpleArm:
    def __init__(self, link_lengths=None):
        # a simple arm with n unit length links connected by hinge joints
        # The base is assumed to be at the origin
        # The joints are assumed to be at the end of each link, with the first joint at the base and the last joint at the end effector
        # even though the end effector is not a joint
        # The axis of rotation for each joint are assumed to be the z-axis
        # The arm is initialized to lie along the x-axis
        
        if link_lengths is None:
            link_lengths = [1, 1, 1]  # Default 3 links of length 1
        self.link_lengths = link_lengths
        self.n = len(link_lengths)
        self.angles = [0]*self.n # joint angles, starting from the base joint to the end effector
        
        # self.Jl is a matrix that contains joints position in local coordinates.
        # Each row contains the coordinates of a joint
        # Number of joints is n+1, including the base and end effector
        self.Jl = np.zeros((self.n+1, 3)) 
        for i in range(1,self.n + 1): # we start from 1 because the base joint is at the origin (0,0,0) and finish the end effector is at the end of the last link
            self.Jl[i,:] = np.array([self.link_lengths[i - 1], 0, 0]) # initialize joint positions to lie along the x-axis

        self.Jw = np.zeros((self.n+1, 3)) # joint positions in world coordinates
        self.FK()

    def FK(self, angles=None): 
        # calculate the forward kinematics of the arm

        # angles is a list of joint angles. If angles is None, the current joint angles are used
        if angles is not None:
            self.angles = angles
        
        # Initial rotation matrix
        Ri = np.eye(3)
        self.Jw[0, :] = np.array([0, 0, 0])  # Base position is always at the origin

        for i in range(1, self.n + 1):
            Ri = Rot(self.angles[i - 1], [0, 0, 1]) @ Ri
            self.Jw[i, :] = Ri @ self.Jl[i, :] + self.Jw[i - 1, :]

        return self.Jw[-1,:] # return the position of the end effector
        
        
    def IK(self, target):
        # calculate the inverse kinematics of the arm
        # target is the position of the end effector in world coordinates
        # return a list of joint angles
        pass
    
    def VelocityJacobian(self, angles=None):
        # calculate the velocity jacobian of the arm
        # return a 3x3 numpy array
        pass

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05)
        for i in range(1,self.n+1):
            vd_arm += vd.Cylinder(pos = [self.Jw[i-1,:], self.Jw[i,:]], r=0.02)
            vd_arm += vd.Sphere(pos = self.Jw[i,:], r=0.05)
        return vd_arm
        

#%%
activeJoint = 0
IK_target = [1,1,0]
def OnSliderAngle(widget, event):
    global activeJoint
    arm.angles[activeJoint] = widget.value
    arm.FK()
    plt.remove("Assembly")
    plt.add(arm.draw())
    plt.render()

def OnCurrentJoint(widget, event):
    global activeJoint
    activeJoint = round(widget.value)
    sliderAngle.value = arm.angles[activeJoint]


def LeftButtonPress(evt):
    global IK_target
    picked = evt.picked3d
    if picked is not None:
        IK_target = picked
        plt.remove("Sphere")
        plt.add(vd.Sphere(pos=IK_target, r=0.05, c='b'))
        plt.render()
    else:
        print("No valid point picked — click on the plane or geometry.")


arm = SimpleArm(link_lengths=[1, 0.5, 1.2, 0.5,2]) 
plt = vd.Plotter()
plt += arm.draw()
plt += vd.Sphere(pos = IK_target, r=0.05, c='b').draggable(True)
plt += vd.Plane(s=[2.1*arm.n,2.1*arm.n]) # a plane to catch mouse events
sliderCurrentJoint = plt.add_slider(OnCurrentJoint, 0, arm.n-1, 0, title="Current joint", pos=3, delayed=True)
sliderAngle =  plt.add_slider(OnSliderAngle,-np.pi,np.pi,0., title="Joint Angle", pos=4)
plt.add_callback('LeftButtonPress', LeftButtonPress) # add Mouse callback
plt.user_mode('2d').show(zoom="tightest")

plt.close()

# %%
