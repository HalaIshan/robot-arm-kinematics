#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np

#%% class for a robot arm

# Define joint_colors BEFORE the class/methods that rely on it.
joint_colors = ["gray", "red", "green", "blue", "orange"]
MODE = 'JT'  # 'JT' = Jacobian-Transpose, 'GN' = Gauss–Newton



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
    def __init__(self,  link_lengths=None):
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
        


    def preprocess_target(self, target):
        # Clip unreachable
        max_reach = sum(self.link_lengths)
        d = np.linalg.norm(target[:2])
        if d > max_reach:
            target[:2] *= (max_reach - 1e-3)/d
        # Break singular
        if all(abs(abs(a)-np.pi)<1e-6 or abs(a)<1e-6 for a in self.angles):
            self.angles[0] += 1e-3
        return target
  
    def IK_JT(self, target):
        # calculate the inverse kinematics of the arm
        # target is the position of the end effector in world coordinates
        # return a list of joint angles

        print(">>> Running Jacobian-Transpose IK <<<")
        target = self.preprocess_target(target.copy())
        max_iters = 1000
        tol       = 1e-3
        alpha     = 0.1

        target = np.array(target)

        max_reach = sum(self.link_lengths)
        d = np.linalg.norm(target[:2])
        if d > max_reach:
            target[:2] *= (max_reach - 1e-3) / d

        if all(abs(abs(a) - np.pi) < 1e-6 or abs(a) < 1e-6 for a in self.angles):
            self.angles[0] += 1e-3

        prev_err = np.inf
        for it in range(max_iters):
            p_end = self.FK()
            error = target[:2] - p_end[:2]
            err_norm = np.linalg.norm(error)
            if err_norm < tol:
                print(f"IK converged in {it} iterations (error {err_norm:.4f})")
                break
            if err_norm > prev_err:
                alpha *= 0.5
            prev_err = err_norm
            J2D = self.VelocityJacobian()[:2, :]
            delta_theta = alpha * (J2D.T @ error)

            if np.linalg.norm(delta_theta) < 1e-4:
                print(f"Δθ too small at iter {it}, exiting (error {err_norm:.4f})")
                break
        
            for i in range(self.n):
                self.angles[i] += delta_theta[i]
                self.angles[i] = np.arctan2(np.sin(self.angles[i]),
                                            np.cos(self.angles[i]))
        else:
            print(f"IK did not converge (final error {err_norm:.4f})")

        return self.angles

    
    def IK_GN(self, target):
        print(">>> Running Gauss-Newton IK <<<")
        target = self.preprocess_target(target.copy())
        max_iters, tol, alpha = 1000, 1e-3, 0.1
        for it in range(max_iters):
            p_end = self.FK()
            error = target[:2] - p_end[:2]
            if np.linalg.norm(error) < tol: break
            J2D = self.VelocityJacobian()[:2,:]
            delta = alpha * (np.linalg.pinv(J2D) @ error)
            if np.linalg.norm(delta) < 1e-4: break
            self.angles = [np.arctan2(np.sin(a+da), np.cos(a+da))
                           for a, da in zip(self.angles, delta)]
    
    def VelocityJacobian(self, angles=None):
        # calculate the velocity jacobian of the arm
        # return a 3x3 numpy array
        if angles is not None:
            self.FK(angles)
        else:
            self.FK()

        z_axis = np.array([0, 0, 1])
        p_end = self.Jw[-1, :]
        J = np.zeros((3, self.n))

        for i in range(self.n):
            p_i = self.Jw[i, :]
            J[:, i] = np.cross(z_axis, p_end - p_i)
            
        return J
    

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05 , c=joint_colors[0])
        for i in range(1,self.n+1):
            vd_arm += vd.Cylinder(pos = [self.Jw[i-1,:], self.Jw[i,:]], r=0.02)
            vd_arm += vd.Sphere(pos = self.Jw[i,:], r=0.05, c=joint_colors[i])
        return vd_arm
    
        

#%%
activeJoint = 0
IK_target = [1,1,0]
def OnSliderAngle(widget, event):
    global activeJoint, arm, plt
    arm.angles[activeJoint] = widget.value
    arm.FK()

    plt.remove("ARM")
    arm_actor = arm.draw()
    arm_actor.name = "ARM"
    plt += arm_actor

    for i in range(arm.n):
        plt.remove(f"JointCol_{i}")

    J = arm.VelocityJacobian()
    p_end = arm.Jw[-1, :]
    scale = 0.3

    for i in range(arm.n):
        col = J[:, i]
        if np.linalg.norm(col) < 1e-8:
            continue

        head = p_end + scale * col
        color = joint_colors[i]
        arrow = vd.Arrow(p_end, head, c=color)
        arrow.name = f"JointCol_{i}"
        plt += arrow

    plt.render()


def OnCurrentJoint(widget, event):
    global activeJoint
    activeJoint = round(widget.value)
    sliderAngle.value = arm.angles[activeJoint]


def LeftButtonPress(evt):
    global IK_target, arm, plt, MODE
    picked = evt.picked3d
    if picked is not None:
        IK_target = np.array(picked)

        # 1) Draw or move the blue target sphere
        plt.remove("Sphere")
        sph = vd.Sphere(pos=IK_target, r=0.05, c='b')
        sph.name = "Sphere"
        plt += sph

        # 2) Run the selected IK solver
        if MODE == 'JT':
            arm.IK_JT(IK_target)
        else:
            arm.IK_GN(IK_target)

        # 3) Recompute FK and redraw the arm
        arm.FK()
        plt.remove("ARM")
        new_arm = arm.draw()
        new_arm.name = "ARM"
        plt += new_arm

        # 4) Remove old Jacobian arrows and draw new ones
        for i in range(arm.n):
            plt.remove(f"JointCol_{i}")

        J = arm.VelocityJacobian()
        p_end = arm.Jw[-1]
        scale = 0.3
        for i in range(arm.n):
            col = J[:, i]
            if np.linalg.norm(col) < 1e-8:
                continue
            head = p_end + scale * col
            arrow = vd.Arrow(p_end, head, c=joint_colors[i % len(joint_colors)])
            arrow.name = f"JointCol_{i}"
            plt += arrow

        plt.render()
    else:
        print("No valid point picked — click on the plane or geometry.")



def OnKeyPress(evt):
    global MODE, mode_label
    # fetch the pressed key symbol from the Vedo interactor
    key = evt.keypress.lower()
    if key == 'j':
        MODE = 'JT'
    elif key == 'n':
        MODE = 'GN'
    mode_label.text(f"Mode: {MODE}  (J=Jacobian, N=Newton)")
    plt.render() 




arm = SimpleArm(link_lengths=[1, 1, 1, 1]) 
plt = vd.Plotter()
arm_actor = arm.draw()
arm_actor.name = "ARM"
plt += arm_actor

# ─── 2) Draw initial arrows, all anchored at p_end ────────────────────────────
J_init = arm.VelocityJacobian()
p_end = arm.Jw[-1, :]
scale = 0.3

for i in range(arm.n):
    col = J_init[:, i]
    if np.linalg.norm(col) < 1e-8:
        continue

    head = p_end + scale * col
    color = joint_colors[i]
    arrow = vd.Arrow(p_end, head, c=color)
    arrow.name = f"JointCol_{i}"
    plt += arrow
# ───────────────────────────────────────────────────────────────────────────────

sph = vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True)
sph.name = "Sphere"
plt += sph
mode_label = vd.Text2D(f"Mode: {MODE}  (J=Jacobian, N=Newton)",pos='top-left', c='black', bg='white')
plt += mode_label
plt += vd.Plane(s=[2.1*arm.n,2.1*arm.n]) # a plane to catch mouse events
sliderCurrentJoint = plt.add_slider(OnCurrentJoint, 0, arm.n-1, 0, title="Current joint", pos=3, delayed=True)
sliderAngle =  plt.add_slider(OnSliderAngle,-np.pi,np.pi,0., title="Joint Angle", pos=4)
plt.add_callback('LeftButtonPress', LeftButtonPress) # add Mouse callback
plt.add_callback('KeyPress', OnKeyPress)
plt.user_mode('2d').show(zoom="tightest")

plt.close()

# %% 