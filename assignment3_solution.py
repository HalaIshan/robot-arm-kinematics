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
    def __init__(self, n=3):
        # a simple arm with n unit length links connected by hinge joints
        # The base is assumed to be at the origin
        # The joints are assumed to be at the end of each link, with the first joint at the base and the last joint at the end effector
        # even though the end effector is not a joint
        # The axis of rotation for each joint are assumed to be the z-axis
        # The arm is initialized to lie along the x-axis
        
        self.n = n # number of links
        self.angles = [0]*self.n # joint angles, starting from the base joint to the end effector
        
        # self.Jl is a matrix that contains joints position in local coordinates.
        # Each row contains the coordinates of a joint
        # Number of joints is n+1, including the base and end effector
        self.Jl = np.zeros((self.n+1, 3)) 
        for i in range(1,n+1): # we start from 1 because the base joint is at the origin (0,0,0) and finish the end effector is at the end of the last link
            self.Jl[i,:] = np.array([1, 0, 0]) # initialize joint positions to lie along the x-axis

        self.Jw = np.zeros((self.n+1, 3)) # joint positions in world coordinates
        self.FK()

        self.vd_arm = vd.Assembly() # empty vedo assembly object to draw the arm

    def FK(self, angles=None): 
        # calculate the forward kinematics of the arm

        # angles is a list of joint angles. If angles is None, the current joint angles are used
        if angles is not None:
            self.angles = angles
        
        # Ri = np.eye(3)
        # for i in range(1,self.n+1): # we start from 1 because the base joint is always at the origin (0,0,0)
        #     Ri = Rot(self.angles[i-1], [0, 0, 1]) @ Ri # The rotation matrix for the i-th joint is the product of all previous joint rotations
            
        #     # The position of the i-th joint in world coordinates is the position of the i-th joint in
        #     # local coordinates rotated by Ri and then translated by the position of the previous joint
        #     self.Jw[i,:] = Ri @ self.Jl[i,:] + self.Jw[i-1,:] 
        
        # Initial rotation matrix
        Ri = np.eye(3)

        # First iteration (i=1)
        Ri_1 = Rot(self.angles[0], [0, 0, 1]) @ Ri  # Compute the rotation matrix for the first joint
        self.Jw[1, :] = Ri_1 @ self.Jl[1, :] + self.Jw[0, :]  # Update the position of the first joint in world coordinates

        # Second iteration (i=2)
        Ri_2 = Rot(self.angles[1], [0, 0, 1]) @ Ri_1  # Compute the rotation matrix for the second joint
        self.Jw[2, :] = Ri_2 @ self.Jl[2, :] + self.Jw[1, :]  # Update the position of the second joint in world coordinates

        return self.Jw[-1,:] # return the position of the end effector
        
    def draw(self):
        self.vd_arm = vd.Assembly()
        self.vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05)
        for i in range(1,self.n+1):
            self.vd_arm += vd.Cylinder(pos = [self.Jw[i-1,:], self.Jw[i,:]], r=0.02)
            self.vd_arm += vd.Sphere(pos = self.Jw[i,:], r=0.05)
        return self.vd_arm
        

#%%

def OnSliderAngle(widget, event):
    arm.angles[1] = widget.value
    arm.FK()
    plt.remove(plt.actors[0])
    plt.add(arm.draw())
    plt.render()

arm = SimpleArm(2)
plt = vd.Plotter()
plt += arm.draw()
plt.add_slider(OnSliderAngle,0.,2*np.pi,0., title="Angle")
plt.show()

plt.close()

# %%
