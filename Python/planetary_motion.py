import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os
import imageio
def three_body_problem(t, z, G, m1, m2, m3):
    # z[0] = x1, z[1] = x1', z[2] = y1, z[3] = y1',
    # z[4] = x2, z[5] = x2', z[6] = y2, z[7] = y2',
    # z[8] = x3, z[9] = x3', z[10] = y3, z[11] = y3',
    x1 = z[0:3:2]
    x2 = z[4:7:2]
    x3 = z[8:11:2]
    r1 = np.linalg.norm(x1 - x2, ord=2)**3
    r2 = np.linalg.norm(x2 - x3, ord=2)**3
    r3 = np.linalg.norm(x1 - x3, ord=2)**3
    z_dot = [None]*z.shape[0]
    # x1
    z_dot[0] = z[1]
    z_dot[1] = -G*m2*(z[0] - z[4])/r1 - G*m2*(z[0] - z[8])/r3
    # y1
    z_dot[2] = z[3]
    z_dot[3] = -G*m2*(z[1] - z[5])/r1 - G*m2*(z[1] - z[9])/r3
    # x2
    z_dot[4] = z[5]
    z_dot[5] = -G*m3*(z[4] - z[8])/r2 - G*m1*(z[4] - z[0])/r1
    # y2
    z_dot[6] = z[7]
    z_dot[7] = -G*m3*(z[6] - z[10])/r2 - G*m1*(z[6] - z[2])/r1
    # x3
    z_dot[8] = z[9]
    z_dot[9] = -G*m1*(z[8] - z[0])/r3 - G*m2*(z[8] - z[4])/r2
    # y3
    z_dot[10] = z[11]
    z_dot[11] = -G*m1*(z[10] - z[2])/r3 - G*m2*(z[10] - z[6])/r2
    return np.array(z_dot)
def circular_orbit(r, t):
    return [r*np.cos(t), r*np.sin(t)]
def circular_veloc(r, t):
    return [-r*np.sin(t), r*np.cos(t)]
if __name__ == "__main__":
    G = 1
    mass_1 = 2
    mass_2 = 2
    mass_3 = 2
    r1 = 2
    r2 = 2
    r3 = 2
    t1 = 0
    t2 = 2*np.pi/3
    t3 = 4*np.pi/3
    #x_init = np.random.randint(-100, 100, size=(6, )) #np.array([-1, 1, 2, 3, 5, -6])
    #v_init = np.random.uniform(-.5, .5, size=(6, ))
    x_init = np.array([*circular_orbit(r1, t1), *circular_orbit(r2, t2), *circular_orbit(r3, t3)])
    v_init = np.array([*circular_veloc(r1, t1), *circular_veloc(r2, t2), *circular_veloc(r3, t3)])
    z_init = np.zeros(shape=(x_init.size+x_init.size, ))
    z_init[::2] = x_init
    z_init[1::2] = v_init
    t0 = 0
    tf = 120
    num_t_steps = 1001
    times = np.linspace(t0, tf, num_t_steps)
    result = integrate.solve_ivp(three_body_problem, (t0, tf), z_init, method='RK45', 
                                 t_eval=times, vectorized=True, args=(G, mass_1, mass_2, mass_3)).y.T
    positions = result[:, ::2]
    velocitys = result[:, 1::2]
    traj1 = positions[:, :2]
    traj2 = positions[:, 2:4]
    traj3 = positions[:, 4:]
    #real_x_excess = .1*np.ptp(real_traj[:, 0])
    #real_y_excess = .1*np.ptp(real_traj[:, 1])
    #simulated_x_excess = .1*np.ptp(simulated_traj[:, 0])
    #simulated_y_excess = .1*np.ptp(simulated_traj[:, 1])
    system_images_folder = f".\\fun_motion"
    if not os.path.exists(system_images_folder):
        os.makedirs(system_images_folder)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(*traj1[0], 'bo', ms=12, label="Initial Condition")
    ax.plot(*traj2[0], 'ro', ms=12, label="Initial Condition")
    ax.plot(*traj3[0], 'go', ms=12, label="Initial Condition")
    ax.set_title("Fun Motion", size=25)
    ax.set_xlabel("$x$", size=15)
    ax.set_ylabel("$y$", size=15)
    #ax.set_xlim(real_traj[:, 0].min() - real_x_excess, real_traj[:, 0].max() + real_x_excess)
    #ax.set_ylim(real_traj[:, 1].min() - real_y_excess, real_traj[:, 1].max() + real_y_excess)
    ax.legend(loc='best', framealpha=1)
    ax.grid(True, which='both')
    fig.tight_layout()
    image_list = [None]*traj1.shape[0]
    gif_duration = 10.
    trail = 20
    for ii in range(traj1.shape[0]):
        offset = min((ii, trail))
        lines1 = ax.plot(*traj1[ii-offset: ii+1].T, 'b--')
        lines2 = ax.plot(*traj2[ii-offset: ii+1].T, 'r--')
        lines3 = ax.plot(*traj3[ii-offset: ii+1].T, 'g--')
        point1 = ax.plot(*traj1[ii].T, 'b*', label="Mass 1")
        point2 = ax.plot(*traj2[ii].T, 'r*', label="Mass 2")
        point3 = ax.plot(*traj3[ii].T, 'g*', label="Mass 3")
        file_name = f"{system_images_folder}\\fun_motion_{ii:05d}.jpg"
        fig.savefig(file_name)
        image_list[ii] = imageio.imread(file_name)
        lines1.pop().remove()
        lines2.pop().remove()
        lines3.pop().remove()
        point1.pop().remove()
        point2.pop().remove()
        point3.pop().remove()
        if ii % 100:
            try:
                os.remove(file_name)
            except:
                pass
    plt.close(fig)
    imageio.mimsave(f"{system_images_folder}\\fun_motion.gif", image_list, duration=gif_duration/len(image_list))