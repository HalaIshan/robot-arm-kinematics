#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np

#%% class for a robot arm

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
        
        # ————— NEW LINES —————
        self.alpha = 0.1     # step‐size for all IK methods
        self.lam   = 0.1    # damping factor for DLS
        # ————————————————

        if link_lengths is None:
            link_lengths = [1, 1, 1] 
        self.link_lengths = link_lengths
        self.n = len(link_lengths)
        self.angles = [0]*self.n 

        self.Jl = np.zeros((self.n+1, 3)) 
        for i in range(1,self.n + 1):
            self.Jl[i,:] = np.array([self.link_lengths[i - 1], 0, 0]) 
        self.Jw = np.zeros((self.n+1, 3)) 
        self.FK()

    def FK(self, angles=None): 
        if angles is not None:
            self.angles = angles
        
        Ri = np.eye(3)
        self.Jw[0, :] = np.array([0, 0, 0]) 

        for i in range(1, self.n + 1):
            Ri = Rot(self.angles[i - 1], [0, 0, 1]) @ Ri
            self.Jw[i, :] = Ri @ self.Jl[i, :] + self.Jw[i - 1, :]

        return self.Jw[-1,:] 
        


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
            if self.n > 2:                  # only apply DLS when redundant
                JJt   = J2D @ J2D.T
                A     = JJt + (self.lam**2)*np.eye(2)
                invA  = np.linalg.inv(A)
                delta = self.alpha * (J2D.T @ (invA @ error))
            else:
                delta = self.alpha * (J2D.T @ error)


            if np.linalg.norm(delta) < 1e-4:
                print(f"Δθ too small at iter {it}, exiting (error {err_norm:.4f})")
                break
        
            for i in range(self.n):
                self.angles[i] = np.arctan2(np.sin(self.angles[i] + delta[i]), np.cos(self.angles[i] + delta[i]))
        else:
            print(f"IK did not converge (final error {err_norm:.4f})")

        self.angles = [np.arctan2(np.sin(a + d), np.cos(a + d))
                for a, d in zip(self.angles, delta)]
        return self.angles

    
    def IK_GN(self, target):
        print(">>> Running Gauss-Newton / DLS IK <<<")
        target = self.preprocess_target(target.copy())
        max_iters, tol = 1000, 1e-3

        for it in range(max_iters):
            p_end   = self.FK()
            error   = target[:2] - p_end[:2]
            err_norm = np.linalg.norm(error)
            if err_norm < tol:
                print(f"DLS-GN converged in {it} iters (err {err_norm:.4f})")
                break

            J2D = self.VelocityJacobian()[:2, :]      # 2×n

            if self.n > 2:
                # ——— Damped least-squares update ———
                JJt      = J2D @ J2D.T                # 2×2
                A        = JJt + (self.lam**2)*np.eye(2)
                invA     = np.linalg.inv(A)
                delta    = self.alpha * (J2D.T @ (invA @ error))
            else:
                # fallback to plain pseudo-inverse when not redundant
                delta    = self.alpha * (np.linalg.pinv(J2D) @ error)

            if np.linalg.norm(delta) < 1e-4:
                print(f"Δθ too small at iter {it}")
                break

            # apply Δθ and wrap into [−π,π]
            for i in range(self.n):
                a = self.angles[i] + delta[i]
                self.angles[i] = np.arctan2(np.sin(a), np.cos(a))
        else:
            print(f"DLS-GN did not converge (final err {err_norm:.4f})")

        return self.angles


            

    def init_IK(self, target, mode):
        """Prepare a manual‐step run."""
        import numpy as np
        self.target     = self.preprocess_target(np.array(target))
        self.mode       = mode         # 'JT' or 'GN'
        self.iter_count = 0
        self.prev_err   = np.inf
        self.alpha      = 0.1

    def step_IK(self):
        """Perform exactly one JT/GN iteration. Returns True when done."""
        tol = 1e-3
        p_end    = self.FK()
        error    = self.target[:2] - p_end[:2]
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            print(f"{self.mode} converged in {self.iter_count} steps")
            return True

        J2D = self.VelocityJacobian()[:2, :]
        if self.mode == 'JT':
            delta = self.alpha * (J2D.T @ error)
        else:  # 'GN'
            delta = self.alpha * (np.linalg.pinv(J2D) @ error)

        if np.linalg.norm(delta) < 1e-4:
            print(f"{self.mode} Δθ too small at step {self.iter_count}")
            return True

        self.angles = [
            np.arctan2(np.sin(a+da), np.cos(a+da))
            for a, da in zip(self.angles, delta)
        ]
        self.prev_err   = err_norm
        self.iter_count += 1
        return False

    
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

def remove_jacobian_arrows(plt, n):
    for i in range(n):
        plt.remove(f"JointCol_{i}")

def draw_jacobian_arrows(arm, plt, scale=0.3):
    J     = arm.VelocityJacobian()
    p_end = arm.Jw[-1]
    for i in range(arm.n):
        col = J[:, i]
        if np.linalg.norm(col) < 1e-8:
            continue
        arrow = vd.Arrow(p_end, p_end + scale*col,
                         c=joint_colors[i % len(joint_colors)])
        arrow.name = f"JointCol_{i}"
        plt += arrow

def reset_label():
    if MODE == 'GN':
        base = f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)\nStep: {arm.iter_count}"
    else:
        base = f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)"
    
    if arm.n > 2:
        extra = "\nUnique: Damped least-squares"
    else:
        extra = "\nUnique: none"
    
    mode_label.text(base + extra)



def OnSliderAngle(widget, event):
    global activeJoint, arm, plt
    arm.angles[activeJoint] = widget.value
    arm.FK()

    plt.remove("ARM")
    arm_actor = arm.draw()
    arm_actor.name = "ARM"
    plt += arm_actor

    remove_jacobian_arrows(plt, arm.n)
    if MODE == 'JT':
        draw_jacobian_arrows(arm, plt)
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

        plt.remove("Sphere")
        sph = vd.Sphere(pos=IK_target, r=0.05, c='b')
        sph.name = "Sphere"
        plt += sph

        if MODE == 'JT':
            arm.IK_JT(IK_target)
        elif MODE == 'GN':
            arm.init_IK(IK_target, 'GN')

        arm.FK()
        plt.remove("ARM")
        new_arm = arm.draw()
        new_arm.name = "ARM"
        plt += new_arm

        remove_jacobian_arrows(plt, arm.n)

        if MODE == 'JT':
            draw_jacobian_arrows(arm, plt)

        reset_label()
        plt.render()

    else:
        mode_label.text(mode_label.text() +"\n⚠️ No valid point picked — click on the plane or geometry.")



def OnKeyPress(evt):
    global MODE, mode_label, arm, plt
    x=0
    key = evt.keypress.lower()
    if key == 'j':
        MODE = 'JT'
        arm.iter_count = 0
        arm.prev_err   = np.inf
        reset_label()
    elif key == 'n':
        MODE = 'GN'
        arm.iter_count = 0
        arm.prev_err   = np.inf
        reset_label()
    elif key == 's':
        if not hasattr(arm, 'target'):
            # No GN target yet — warn user in the label
            mode_label.text(
                f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)\n"
                "⚠️ Click to set a GN target first."
            )
            return
        if MODE == 'GN':
            done = arm.step_IK()
            if done:
                mode_label.text(
                    f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)\n"
                    f"✅ GN converged in {arm.iter_count} steps")
                return  
            arm.FK()
            plt.remove("ARM")
            new_arm = arm.draw()
            new_arm.name = "ARM"
            plt += new_arm
            remove_jacobian_arrows(plt, arm.n)
            reset_label()
            plt.render()
        else:
            mode_label.text(
                f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)\n"
                "Press 'n' to switch to GN before stepping.")
    else:
        reset_label()    
    plt.render() 




arm = SimpleArm(link_lengths=[1, 1, 1, 1]) 
plt = vd.Plotter()
arm_actor = arm.draw()
arm_actor.name = "ARM"
plt += arm_actor

if MODE == 'JT':
    remove_jacobian_arrows(plt, arm.n)
    draw_jacobian_arrows(arm, plt)


sph = vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True)
sph.name = "Sphere"
plt += sph
mode_label = vd.Text2D(f"Mode: {MODE}  (J=Jacobian, N=Newton, S=step)", pos='top-left', c='black', bg='white')
plt += mode_label
plt += vd.Plane(s=[2.1*arm.n,2.1*arm.n]) # a plane to catch mouse events
sliderCurrentJoint = plt.add_slider(OnCurrentJoint, 0, arm.n-1, 0, title="Current joint", pos=3, delayed=True)
sliderAngle =  plt.add_slider(OnSliderAngle,-np.pi,np.pi,0., title="Joint Angle", pos=4)
plt.add_callback('LeftButtonPress', LeftButtonPress) # add Mouse callback
plt.add_callback('KeyPress', OnKeyPress)
plt.user_mode('2d').show(zoom="tightest")

plt.close()

# %% 