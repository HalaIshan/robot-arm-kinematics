#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np

#%% class for a robot arm

joint_colors = ["gray", "red", "green", "blue", "orange"]
MODE = 'JT'  # 'JT' = Jacobian-Transpose, 'GN' = Gauss–Newton

def wrap_deg_180(degs):
    # maps any angle to [-180,180)
    return (degs + 180) % 360 - 180

def draw_manipulability_ellipse(arm, plt, scale=0.5, n_samples=100):
    """
    Draw a 2D manipulability ellipse *with joint-limit weighting*.
    Columns of J2D are scaled by w_i in [0,1] measuring proximity to ±limit.
    """
    # 1) remove any existing ellipse
    try: plt.remove("Ellipsoid")
    except: pass
    # 2) compute raw 2×n Jacobian and mask frozen joints
    J2D = arm.VelocityJacobian()[:2, :].copy()
    J2D[:, :activeJoint] = 0

    # 3) build per-joint weights w_i ∈ [0,1]
    weights = []
    for i, θ in enumerate(arm.angles):
        mn, mx = arm.joint_min[i], arm.joint_max[i]
        mid     = 0.5*(mn + mx)
        half    = 0.5*(mx - mn)
        w       = max(0.0, (half - abs(θ - mid))/half)
        weights.append(w)
    W = np.diag(weights)

    # 4) form weighted Jacobian
    Jw2D = J2D @ W

    # 5) SVD on weighted Jacobian
    U, S, _ = np.linalg.svd(Jw2D, full_matrices=False)
    origin = arm.Jw[-1][:2]

    if S[1] < 1e-6:
        # rank-1 → draw a single line along the principal direction
        vec  = U[:,0] * S[0] * scale
        line = vd.Line([[ *origin, 0 ], [ *(origin + vec), 0 ]])
        line.c("purple").lw(3).name = "Ellipsoid"
        plt += line
        return

    # otherwise (rank-2) fall back to your normal ellipse‐sampling:
    pts = []
    for t in np.linspace(0, 2*np.pi, n_samples):
        local = np.array([S[0]*np.cos(t), S[1]*np.sin(t)]) * scale
        world = origin + U @ local
        pts.append([ *world, 0 ])
    ellipse = vd.Line(pts, closed=True).c("purple").alpha(0.4).lw(2)
    ellipse.name = "Ellipsoid"
    plt += ellipse


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
    def __init__(self,  link_lengths=None, joint_types=None):
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
        if joint_types is None:
            joint_types = ['R'] * len(link_lengths)
        self.link_lengths = link_lengths
        self.joint_types = joint_types         # list of 'R' or 'P'
        self.link_params = link_lengths        # for R: link length; for P: max extension
        self.n = len(link_lengths)
        self.angles = [0]*self.n 

        # joint limits in same units:
        self.joint_min   = [-np.pi/4 if jt=='R' else -lp
                        for jt,lp in zip(self.joint_types, self.link_params)]
        self.joint_max   = [ np.pi/4 if jt=='R' else  lp
                        for jt,lp in zip(self.joint_types, self.link_params)]

        # ————— DISABLE limits for now (full 360°) —————
        #self.joint_min = [-2*np.pi] * self.n
        #self.joint_max = [ 2*np.pi] * self.n
        # ——————————————————————————————————————

        self.Jl = np.zeros((self.n+1, 3)) 
        for i in range(1,self.n + 1):
            self.Jl[i,:] = np.array([self.link_lengths[i - 1], 0, 0]) 
        self.Jw = np.zeros((self.n+1, 3)) 
        self.FK()

    def FK(self, angles=None):
        if angles is not None:
            self.angles = angles
        Ri = np.eye(3)
        self.Jw[0,:] = np.zeros(3)
        for i in range(self.n):
            jt = self.joint_types[i]
            val = self.angles[i]
            if jt == 'R':
                Ri = Rot(val, [0,0,1]) @ Ri
                disp = self.link_params[i]
            else:  # 'P'
                # no extra rotation
                disp = val              # linear displacement
            # translate along local x-axis by disp:
            translation = Ri @ np.array([disp,0,0])
            self.Jw[i+1,:] = self.Jw[i,:] + translation
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
        alpha2     = 0.1

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
                alpha2 *= 0.5
            prev_err = err_norm
            J2D = self.VelocityJacobian()[:2, :]             
            # —————————————————————————————————————————————————————————
            #  modified IK_JT snippet — freeze first `activeJoint` DOFs
            free_joints = self.n - activeJoint
            if free_joints > 2:
                JJt   = J2D @ J2D.T
                A     = JJt + (self.lam**2)*np.eye(2)
                invA  = np.linalg.inv(A)
                delta = alpha2 * (J2D.T @ (invA @ error))
            else:
                delta = alpha2 * (J2D.T @ error)

            # ❄️ freeze the first `activeJoint` joints:
            delta[:activeJoint] = 0

            
            # apply, wrap, clamp & FK-update per joint
            for i in range(self.n):
                a = self.angles[i] + delta[i]
                w = np.arctan2(np.sin(a), np.cos(a))
                # clamp to [joint_min, joint_max]
                self.angles[i] = np.clip(w,
                                        self.joint_min[i],
                                        self.joint_max[i])
                # update forward kinematics so joint i+1 sees this
                self.FK()


            # —————————————————————————————————————————————————————————

        else:
            print(f"IK did not converge (final error {err_norm:.4f})")
        draw_manipulability_ellipse(self, plt, scale=0.5)
        return self.angles

    
    def IK_GN(self, target):
        print(">>> Running Gauss-Newton / DLS IK <<<")
        target = self.preprocess_target(target.copy())
        max_iters, tol = 1000, 1e-3
        prev_err = np.inf

        for it in range(max_iters):
            p_end   = self.FK()
            error   = target[:2] - p_end[:2]
            err_norm = np.linalg.norm(error)
            # 1) Converged?
            if err_norm < tol:
                print(f"GN converged in {it} iterations (error {err_norm:.4f})")
                break

            # 2) Stalled?
            if abs(prev_err - err_norm) < 1e-6:
                print(f"GN stalled after {it} iterations (error {err_norm:.4f})")
                break

            prev_err = err_norm
            # 3) Compute delta…
            J2D = self.VelocityJacobian()[:2, :]      # 2×n
            free_joints = self.n - activeJoint
            if free_joints > 2:
                JJt   = J2D @ J2D.T
                A     = JJt + (self.lam**2)*np.eye(2)
                invA  = np.linalg.inv(A)
                delta = self.alpha * (J2D.T @ (invA @ error))
            else:
                delta = self.alpha * (np.linalg.pinv(J2D) @ error)

            delta[:activeJoint] = 0

            # 4) Tiny‐step?
            if np.linalg.norm(delta) < 1e-6:
                print(f"GN Δθ too small after {it} iterations (error {err_norm:.4f})")
                break

            # 5) Apply it
            for i in range(self.n):
                a = self.angles[i] + delta[i]
                w = np.arctan2(np.sin(a), np.cos(a))
                self.angles[i] = np.clip(w,
                                        self.joint_min[i],
                                        self.joint_max[i])
                self.FK()  

        else:
            print(f"DLS-GN did not converge (final err {err_norm:.4f})")
        
        # draw ellipse exactly once at the final pose
        draw_manipulability_ellipse(self, plt, scale=0.5)
        return self.angles


            

    def init_IK(self, target, mode):
        """Prepare a manual‐step run."""
        global activeJoint, plt
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
            draw_manipulability_ellipse(self, plt, scale=0.5)
            return True

        J2D = self.VelocityJacobian()[:2, :]
        if self.mode == 'JT':
            delta = self.alpha * (J2D.T @ error)
        else:  # 'GN'
            delta = self.alpha * (np.linalg.pinv(J2D) @ error)

        if np.linalg.norm(delta) < 1e-4:
            print(f"{self.mode} Δθ too small at step {self.iter_count}")
            draw_manipulability_ellipse(self, plt, scale=0.5)
            return True
        
        delta[:activeJoint] = 0

        for i in range(self.n):
            w = np.arctan2(np.sin(self.angles[i] + delta[i]),
                           np.cos(self.angles[i] + delta[i]))
            # clamp
            w = np.clip(w, self.joint_min[i], self.joint_max[i])
            # write back & update FK so joint i+1 uses this
            self.angles[i] = w
            self.FK()


        self.prev_err   = err_norm
        self.iter_count += 1
        return False

    
    def VelocityJacobian(self):
        # ensure self.Jw is up-to-date
        self.FK()
        z     = np.array([0,0,1])
        p_end = self.Jw[-1,:]
        J     = np.zeros((3,self.n))
        # also track Ri per joint for prismatic axes
        Ri = np.eye(3)
        for i in range(self.n):
            jt = self.joint_types[i]
            if jt == 'R':
                # revolute: axis z
                p_i = self.Jw[i,:]
                J[:,i] = np.cross(z, p_end - p_i)
                # update Ri for next link
                Ri = Rot(self.angles[i], [0,0,1]) @ Ri
            else:
                # prismatic: translation along local x-axis
                axis = Ri @ np.array([1,0,0])
                J[:,i] = axis
                # Ri unchanged
        return J

    

    def draw(self):
        global joint_colors
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05 , c=joint_colors[0])
        for i in range(1,self.n+1):
            p0 = self.Jw[i-1]
            p1 = self.Jw[i]
            if np.linalg.norm(p1 - p0) > 1e-6:
                vd_arm += vd.Cylinder(pos=[p0, p1], r=0.02)
            vd_arm += vd.Sphere(pos=p1, r=0.05, c=joint_colors[i])
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
    # how many joints remain free (not frozen by the slider)
    free_joints = arm.n - activeJoint
    # decide if we’re in the redundant, unique‐solution regime
    if free_joints > 2:
        status = "using DLS (unique solution)"
    else:
        status = "no redundancy"

    # base mode & status line
    base = (f"Mode: {MODE} \n [{status}, {free_joints} free joint"
            f"{'s' if free_joints != 1 else ''}]\n"
            "(J=Jacobian, N=Newton, S=step)")

    # append step count for GN
    if MODE == 'GN':
        txt = base + f"\nStep: {arm.iter_count}"
    else:
        txt = base

    mode_label.text(txt)





def OnSliderAngle(widget, event):
    global activeJoint, arm, plt
    try:
        plt.remove("Ellipsoid")
    except:
        pass
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
    global activeJoint, sliderAngle, arm, plt

    # 1) Which joint did we just select?
    activeJoint = int(round(widget.value))
    jt = arm.joint_types[activeJoint]

    # 2) Grab the VTK slider repre­sentation
    rep = sliderAngle.GetRepresentation()

    # 3) If it’s revolute, keep the old [-π, π] range
    if jt == 'R':
        rep.SetMinimumValue(-np.pi)
        rep.SetMaximumValue( np.pi)

    # 4) If it’s prismatic, switch to [–L, L]
    else:
        L = arm.link_params[activeJoint]
        rep.SetMinimumValue(-L)
        rep.SetMaximumValue( L)

    # 5) Finally, move the thumb to the current angle/extension
    sliderAngle.value = arm.angles[activeJoint]

    # 6) Optional: redraw the slider if needed
    plt.render()



def LeftButtonPress(evt):
    global IK_target, arm, plt, MODE, mode_label

    picked = evt.picked3d
    if picked is not None:
        IK_target = np.array(picked)
        # ————— NEW: set per-joint limits ±45° around current angles —————
        for i in range(arm.n):
            base = arm.angles[i]
            arm.joint_min[i] = base - np.pi/4
            arm.joint_max[i] = base + np.pi/4
        # ————————————————————————————————————————————————————————

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

    else:
        mode_label.text(mode_label.text() +"\n⚠️ No valid point picked — click on the plane or geometry.")
    plt.render()




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
        try:
            plt.remove("Ellipsoid")
        except:
            pass
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



# 'R' for revolute, 'P' for prismatic
arm = SimpleArm(
    link_lengths=[1, 1, 1, 1],
    joint_types = ['R','R','P','R']
)
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